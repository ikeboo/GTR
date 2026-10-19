"""Export-only GLA lowering with intermediate tensors of rank at most four.

Fold batch/head axes before constructing per-key causal decay matrices. The
recurrence and checkpoint parameters are unchanged; no attention approximation
or gate clipping is introduced. Training continues to use gla_portable.py.
"""
import torch
import torch.nn.functional as F


def coreml_gla(q, k, v, g, initial_state=None, output_final_state=False,
               cu_seqlens=None, chunk_size=32):
    if cu_seqlens is not None:
        raise ValueError('CoreML export requires unpacked fixed-length sequences')
    batch, length, heads, key_dim = q.shape
    value_dim, original_dtype = v.shape[-1], q.dtype
    q, k, v, g = (x.float() for x in (q, k, v, g))
    state = (initial_state.float().reshape(batch * heads, key_dim, value_dim)
             if initial_state is not None else q.new_zeros(batch * heads, key_dim, value_dim))
    outputs = []
    for start in range(0, length, chunk_size):
        stop = min(start + chunk_size, length)
        steps = stop - start
        qc, kc, gates = (x[:, start:stop].permute(0, 2, 3, 1).reshape(
            batch * heads, key_dim, steps) for x in (q, k, g))
        vc = v[:, start:stop].transpose(1, 2).reshape(batch * heads, steps, value_dim)
        gates = gates.cumsum(dim=-1)
        decay = (gates.unsqueeze(-1) - gates.unsqueeze(-2)).clamp_max(0).exp()
        scores = (qc.unsqueeze(-1) * kc.unsqueeze(-2) * decay).sum(dim=1)
        causal = torch.ones(steps, steps, dtype=torch.bool, device=q.device).tril_()
        scores = scores.masked_fill(~causal, 0)
        from_state = torch.bmm((qc * gates.exp()).transpose(1, 2), state)
        within = torch.bmm(scores, vc)
        outputs.append(((from_state + within) * key_dim ** -0.5).reshape(
            batch, heads, steps, value_dim).transpose(1, 2))
        end_gate = gates[:, :, -1]
        weighted_keys = kc * (end_gate.unsqueeze(-1) - gates).exp()
        state = state * end_gate.exp().unsqueeze(-1) + torch.bmm(weighted_keys, vc)
    result = torch.cat(outputs, dim=1).to(original_dtype)
    return result, state.reshape(batch, heads, key_dim, value_dim) if output_final_state else None


def parallel_coreml_gla(q, k, v, g, initial_state=None, output_final_state=False,
                        cu_seqlens=None, chunk_size=32, direct_gate_sums=False):
    """Batch within-chunk work and evaluate carried states by associative scan.

    Each chunk maps S to A*S+Z. Composition is associative, so a logarithmic
    prefix scan replaces the Python-unrolled sequential chunk recurrence.
    Prefix gates use a constant triangular matrix instead of cumsum. All
    arithmetic tensors have rank at most four. Padding has zero keys/values
    and zero log decay, so it preserves the final state.
    """
    if cu_seqlens is not None:
        raise ValueError('CoreML export requires unpacked fixed-length sequences')
    batch, length, heads, key_dim = q.shape
    value_dim, original_dtype = v.shape[-1], q.dtype
    chunks = (length + chunk_size - 1) // chunk_size
    padded_length = chunks * chunk_size
    values = [x.float() for x in (q, k, v, g)]
    if padded_length != length:
        values = [F.pad(x, (0, 0, 0, 0, 0, padded_length - length)) for x in values]
    q, k, v, g = values
    qc, kc, gates = (x.permute(0, 2, 1, 3).reshape(batch * heads * chunks, chunk_size, key_dim)
                    .transpose(1, 2) for x in (q, k, g))
    vc = v.permute(0, 2, 1, 3).reshape(batch * heads * chunks, chunk_size, value_dim)
    triangle = torch.ones(chunk_size, chunk_size, device=q.device).triu()
    local_gates = gates
    gates = torch.matmul(local_gates, triangle)
    if direct_gate_sums:
        # Avoid subtracting two large rounded prefix sums on the ANE. Each
        # causal interval is summed directly using a fixed binary projection.
        indices = torch.arange(chunk_size, device=q.device)
        intervals = ((indices[:, None, None] > indices[None, None, :]) &
                     (indices[:, None, None] <= indices[None, :, None])).float()
        differences = torch.matmul(local_gates, intervals.reshape(chunk_size, -1)).reshape(
            batch * heads * chunks, key_dim, chunk_size, chunk_size)
    else:
        differences = gates.unsqueeze(-1) - gates.unsqueeze(-2)
    decay = differences.clamp_max(0).exp()
    scores = (qc.unsqueeze(-1) * kc.unsqueeze(-2) * decay).sum(dim=1)
    causal = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device).tril_()
    scores = scores.masked_fill(~causal, 0)
    end_gate = gates[:, :, -1:]
    tail_gates = (torch.matmul(local_gates, torch.ones_like(triangle).tril(-1))
                  if direct_gate_sums else end_gate - gates)
    weighted_keys = kc * tail_gates.exp()
    states = torch.bmm(weighted_keys, vc).reshape(batch * heads, chunks, key_dim, value_dim)
    factors = end_gate.exp().reshape(batch * heads, chunks, key_dim, 1)
    initial = (initial_state.float().reshape(batch * heads, 1, key_dim, value_dim)
               if initial_state is not None else q.new_zeros(batch * heads, 1, key_dim, value_dim))
    states = torch.cat((states[:, :1] + factors[:, :1] * initial, states[:, 1:]), dim=1)
    distance = 1
    while distance < chunks:
        states = torch.cat((states[:, :distance],
                            states[:, distance:] + factors[:, distance:] * states[:, :-distance]), dim=1)
        factors = torch.cat((factors[:, :distance],
                             factors[:, distance:] * factors[:, :-distance]), dim=1)
        distance *= 2
    previous = torch.cat((initial, states[:, :-1]), dim=1).reshape(
        batch * heads * chunks, key_dim, value_dim)
    from_state = torch.bmm((qc * gates.exp()).transpose(1, 2), previous)
    within = torch.bmm(scores, vc)
    output = ((from_state + within) * key_dim ** -0.5).reshape(batch, heads, padded_length, value_dim)
    output = output[:, :, :length].transpose(1, 2).to(original_dtype)
    return output, states[:, -1].reshape(batch, heads, key_dim, value_dim) if output_final_state else None
