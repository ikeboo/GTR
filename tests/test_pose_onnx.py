"""Regression checks for pose ONNX gates and tied decoder query scores."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from engine.gtr.backbone.vit_adapter import GatedLinearAttention
from engine.gtr.pose.decoder import select_query_indices


class AttentionOutput(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = GatedLinearAttention(hidden_size=16, num_heads=2)

    def forward(self, images):
        return self.attention(images)[0]


class QuerySelection(torch.nn.Module):
    def forward(self, scores):
        return select_query_indices(scores, 4, deploy=True)


@unittest.skipUnless(importlib.util.find_spec('onnx') and
                     importlib.util.find_spec('onnxruntime'), 'ONNX dependencies unavailable')
class PoseOnnxTest(unittest.TestCase):
    def test_tied_query_scores(self):
        import onnxruntime as ort

        model = QuerySelection().eval()
        scores = torch.tensor([[0.5, 0.9, 0.5, 0.5, 0.5, 0.1]])
        expected = np.array([[1, 0, 2, 3]])
        np.testing.assert_array_equal(model(scores).numpy(), expected)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'queries.onnx'
            torch.onnx.export(model, (scores,), path, input_names=['scores'],
                              opset_version=18, dynamo=False)
            session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
            actual = session.run(None, {'scores': scores.numpy()})[0]
            np.testing.assert_array_equal(actual, expected)

    def test_strong_negative_gates(self):
        import onnxruntime as ort

        torch.manual_seed(7)
        model = AttentionOutput().eval()
        with torch.no_grad():
            model.attention.gk_proj[1].weight.zero_()
            model.attention.gk_proj[1].bias.copy_(torch.linspace(-100, 10, 8))
        model.attention.convert_to_deploy()
        # Cross a chunk boundary, including a final partial chunk.
        inputs = torch.randn(1, 35, 16)
        with tempfile.TemporaryDirectory() as directory, torch.inference_mode():
            path = Path(directory) / 'gla.onnx'
            torch.onnx.export(model, (inputs,), path, input_names=['images'],
                              opset_version=18, dynamo=False)
            session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
            for sample in (inputs, inputs * 2):
                expected = model(sample).numpy()
                actual = session.run(None, {'images': sample.numpy()})[0]
                self.assertTrue(np.isfinite(actual).all())
                np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)


if __name__ == '__main__':
    unittest.main()
