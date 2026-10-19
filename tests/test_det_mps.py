"""Regression checks for detection's distribution integral on CPU and MPS."""

import unittest

import torch

from engine.gtr.det.decoder import Integral


class DetectionIntegralTest(unittest.TestCase):
    def check_device(self, device):
        torch.manual_seed(7)
        reference_logits = torch.randn(2, 13, 4 * 33, requires_grad=True)
        reference_project = torch.linspace(-5, 5, 33, requires_grad=True)
        logits = reference_logits.detach().to(device).requires_grad_()
        project = reference_project.detach().to(device).requires_grad_()
        # Independent CPU expression for the expectation of each distribution.
        probabilities = reference_logits.reshape(2, 13, 4, 33).softmax(dim=-1)
        expected = torch.einsum('bqck,k->bqc', probabilities, reference_project)
        actual = Integral(reg_max=32)(logits, project)
        torch.testing.assert_close(actual.cpu(), expected, atol=2e-5, rtol=2e-5)
        actual.square().mean().backward()
        expected.square().mean().backward()
        for computed, reference in ((logits.grad, reference_logits.grad),
                                    (project.grad, reference_project.grad)):
            self.assertTrue(torch.isfinite(computed).all().item())
            torch.testing.assert_close(computed.cpu(), reference, atol=2e-5, rtol=2e-5)

    def test_integral_cpu(self):
        self.check_device('cpu')

    @unittest.skipUnless(torch.backends.mps.is_available(), 'MPS unavailable')
    def test_integral_mps(self):
        self.check_device('mps')


if __name__ == '__main__':
    unittest.main()
