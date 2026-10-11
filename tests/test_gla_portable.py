"""Numerical checks for the PyTorch GLA path used on MPS and CPU."""

import unittest

import torch

from engine.gtr.backbone.gla_portable import portable_gla


class PortableGLATest(unittest.TestCase):
    def check_device(self, device):
        torch.manual_seed(1)
        q = torch.randn(2, 11, 3, 4, device=device, requires_grad=True)
        k = torch.randn(2, 11, 3, 4, device=device, requires_grad=True)
        v = torch.randn(2, 11, 3, 5, device=device, requires_grad=True)
        gates = torch.nn.functional.logsigmoid(
            torch.randn(2, 11, 3, 4, device=device)) / 16
        output, state = portable_gla(q, k, v, gates,
                                     output_final_state=True, chunk_size=4)

        reference_state = torch.zeros(2, 3, 4, 5, device=device)
        reference_output = []
        for index in range(q.shape[1]):
            reference_state = (gates[:, index].exp().unsqueeze(-1) * reference_state
                               + k[:, index].unsqueeze(-1) * v[:, index].unsqueeze(-2))
            reference_output.append(torch.einsum(
                'bhk,bhkv->bhv', q[:, index], reference_state) / 2)
        reference_output = torch.stack(reference_output, dim=1)

        torch.testing.assert_close(output, reference_output, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(state, reference_state, atol=2e-5, rtol=2e-5)
        output.square().mean().backward()
        for tensor in (q, k, v):
            self.assertTrue(bool(torch.isfinite(tensor.grad).all()))

    def test_cpu(self):
        self.check_device('cpu')

    def check_large_negative_gates(self, device):
        q = torch.randn(1, 32, 2, 4, device=device, requires_grad=True)
        k = torch.randn(1, 32, 2, 4, device=device, requires_grad=True)
        v = torch.randn(1, 32, 2, 4, device=device, requires_grad=True)
        gates = torch.full_like(q, -4.0, requires_grad=True)
        output, _ = portable_gla(q, k, v, gates, chunk_size=32)
        output.square().mean().backward()
        for tensor in (q, k, v, gates):
            self.assertTrue(bool(torch.isfinite(tensor.grad).all()))

    def test_large_negative_gates_cpu(self):
        self.check_large_negative_gates('cpu')

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS unavailable')
    def test_mps(self):
        self.check_device('mps')

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS unavailable')
    def test_large_negative_gates_mps(self):
        self.check_large_negative_gates('mps')


if __name__ == '__main__':
    unittest.main()
