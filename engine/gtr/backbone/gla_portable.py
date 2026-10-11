"""PyTorch implementation of the GLA operations used by the ViT backbone.

This path runs on MPS and CPU, where FLA's Triton kernels are unavailable.
The norm parameters keep FLA's names and shapes for checkpoint loading.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


ACT2FN = {
    'swish': F.silu, 'silu': F.silu, 'sigmoid': torch.sigmoid,
    'relu': F.relu, 'elu': F.elu, 'gelu': F.gelu,
}


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, elementwise_affine=True, eps=1e-5, dtype=None):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype)) if elementwise_affine else None

    def reset_parameters(self):
        if self.weight is not None:
            nn.init.ones_(self.weight)

    def forward(self, x):
        value = x.float()
        value = value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + self.eps)
        if self.weight is not None:
            value = value * self.weight
        return value.to(x.dtype)


class FusedRMSNormGated(RMSNorm):
    def __init__(self, hidden_size, elementwise_affine=True, eps=1e-5, activation='swish'):
        super().__init__(hidden_size, elementwise_affine=elementwise_affine, eps=eps)
        self.activation = activation

    def forward(self, x, gate):
        return super().forward(x) * ACT2FN[self.activation](gate)


class ShortConvolution(nn.Module):
    def __init__(self, hidden_size, kernel_size, bias=False, activation='silu'):
        super().__init__()
        self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size,
                              groups=hidden_size, bias=bias)
        self.kernel_size = kernel_size
        self.activation = ACT2FN[activation]

    def forward(self, x, cache=None, output_final_state=False, cu_seqlens=None):
        if cu_seqlens is not None:
            raise NotImplementedError('Packed short convolution is unavailable on MPS')
        previous = cache if cache is not None else x.new_zeros(x.shape[0], self.kernel_size - 1, x.shape[2])
        padded = torch.cat([previous, x], dim=1)
        out = self.activation(self.conv(padded.transpose(1, 2)).transpose(1, 2))
        new_cache = padded[:, -(self.kernel_size - 1):].detach() if output_final_state else None
        return out, new_cache


def get_unpad_data(mask):
    lengths = mask.to(torch.int32).sum(dim=-1)
    indices = mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
    cu_seqlens = torch.cat([lengths.new_zeros(1), lengths.cumsum(0)])
    return indices, cu_seqlens, int(lengths.max().item())


def index_first_axis(x, indices):
    return x.index_select(0, indices)


def pad_input(x, indices, batch_size, sequence_length):
    padded = x.new_zeros(batch_size * sequence_length, *x.shape[1:])
    padded.index_copy_(0, indices, x)
    return padded.reshape(batch_size, sequence_length, *x.shape[1:])


def _chunk_gla(q, k, v, g, initial_state, output_final_state, chunk_size):
    """Evaluate S_t = exp(g_t) S_(t-1) + k_t v_t^T in short chunks.

    The within-chunk causal attention is vectorized; the carried state only
    crosses chunk boundaries. Differences of cumulative gates are non-positive
    for causal positions, so the exponentials stay bounded.
    """
    batch, length, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    original_dtype = q.dtype
    q, k, v, g = (tensor.float() for tensor in (q, k, v, g))
    state = (initial_state.float() if initial_state is not None else
             q.new_zeros(batch, heads, key_dim, value_dim))
    scale = key_dim ** -0.5
    outputs = []

    for start in range(0, length, chunk_size):
        stop = min(start + chunk_size, length)
        qc = q[:, start:stop].transpose(1, 2)
        kc = k[:, start:stop].transpose(1, 2)
        vc = v[:, start:stop].transpose(1, 2)
        gates = g[:, start:stop].transpose(1, 2).cumsum(dim=2)
        steps = stop - start

        # The upper triangle is discarded below. Its gate differences are
        # positive and can overflow exp(), poisoning backward with 0 * inf.
        decay = (gates.unsqueeze(3) - gates.unsqueeze(2)).clamp_max(0).exp()
        scores = (qc.unsqueeze(3) * kc.unsqueeze(2) * decay).sum(dim=-1)
        causal = torch.ones(steps, steps, dtype=torch.bool, device=q.device).tril_()
        scores = scores.masked_fill(~causal, 0)
        from_state = torch.einsum('bhtk,bhkv->bhtv', qc * gates.exp(), state)
        within_chunk = torch.matmul(scores, vc)
        outputs.append(((from_state + within_chunk) * scale).transpose(1, 2))

        end_gate = gates[:, :, -1]
        weighted_keys = kc * (end_gate.unsqueeze(2) - gates).exp()
        state = state * end_gate.exp().unsqueeze(-1) + torch.einsum(
            'bhsk,bhsv->bhkv', weighted_keys, vc)

    output = torch.cat(outputs, dim=1).to(original_dtype)
    return output, state if output_final_state else None


def portable_gla(q, k, v, g, initial_state=None, output_final_state=False,
                 cu_seqlens=None, chunk_size=32):
    if cu_seqlens is None:
        return _chunk_gla(q, k, v, g, initial_state, output_final_state, chunk_size)

    boundaries = cu_seqlens.cpu().tolist()
    outputs, states = [], []
    for index, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        state = initial_state[index:index + 1] if initial_state is not None else None
        result, final = _chunk_gla(q[:, start:stop], k[:, start:stop], v[:, start:stop],
                                   g[:, start:stop], state, output_final_state, chunk_size)
        outputs.append(result)
        if output_final_state:
            states.append(final)
    return torch.cat(outputs, dim=1), torch.cat(states, dim=0) if states else None
