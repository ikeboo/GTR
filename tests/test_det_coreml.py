"""CoreML lowering regressions and compatibility with the Torch postprocessor."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

import numpy as np
import torch


@unittest.skipUnless(importlib.util.find_spec('coremltools') and sys.platform == 'darwin',
                     'CoreML prediction requires macOS and coremltools')
class DetectionCoreMLTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tools.export_det_coreml import register_conversion_ops
        register_conversion_ops()

    def convert(self, model, sample, precision=None):
        import coremltools as ct
        with torch.no_grad():
            traced = torch.jit.trace(model.eval(), sample, check_trace=False)
        return ct.convert(
            traced, inputs=[ct.TensorType(name='images', shape=sample.shape, dtype=np.float32)],
            outputs=[ct.TensorType(name='result', dtype=np.float32)],
            minimum_deployment_target=ct.target.macOS15,
            compute_units=ct.ComputeUnit.CPU_ONLY,
            compute_precision=precision or ct.precision.FLOAT32,
        )

    def test_vector_linear_weight(self):
        class Integral(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.linear(x, torch.tensor([-2., -0.5, 0., 0.5, 2.]))
        sample = torch.randn(2, 3, 5)
        model = Integral()
        converted = self.convert(model, sample)
        with torch.no_grad():
            expected = model(sample).numpy()
        actual = converted.predict({'images': sample.numpy()})['result']
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)

    def test_ane_projection_convolution_matches_linear(self):
        import tools.export_det_coreml as exporter
        model = torch.nn.Linear(16, 24).eval()
        sample = torch.randn(2, 7, 16)
        previous = exporter._ANE_LAYOUT
        try:
            exporter._ANE_LAYOUT = True
            converted = self.convert(model, sample)
        finally:
            exporter._ANE_LAYOUT = previous
        self.assertTrue(any(op.op_type == 'conv' for op in converted._mil_program.functions['main'].operations))
        with torch.no_grad():
            expected = model(sample).numpy()
        actual = converted.predict({'images': sample.numpy()})['result']
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)

    def test_fp32_normalization_with_fp16_projections(self):
        import coremltools as ct
        from tools.export_det_coreml import fp16_except_norms
        torch.manual_seed(19)
        model = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.LayerNorm(16),
                                    torch.nn.Linear(16, 8)).eval()
        sample = torch.randn(1, 7, 16) * 800
        converted = self.convert(model, sample, ct.transform.FP16ComputePrecision(op_selector=fp16_except_norms))
        operations = converted._mil_program.functions['main'].operations
        self.assertTrue(any(op.op_type == 'linear' and op.outputs[0].dtype.__name__ == 'fp16'
                            for op in operations))
        self.assertTrue(any(op.op_type == 'layer_norm' and op.outputs[0].dtype.__name__ == 'fp32'
                            for op in operations))
        with torch.no_grad():
            expected = model(sample).numpy()
        actual = converted.predict({'images': sample.numpy()})['result']
        np.testing.assert_allclose(actual, expected, atol=0.002, rtol=0)

    def test_exp_silu_is_finite_for_extreme_inputs(self):
        import coremltools as ct
        import tools.export_det_coreml as exporter
        class Activation(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.silu(x)
        sample = torch.tensor([[-1000., -100., -20., -10., -5., -2., -1.,
                                0., 1., 2., 5., 10., 100., 1000.]])
        previous = exporter._SILU_EXP
        try:
            exporter._SILU_EXP = True
            converted = self.convert(Activation(), sample, ct.precision.FLOAT16)
        finally:
            exporter._SILU_EXP = previous
        self.assertFalse(any(op.op_type == 'silu' for op in converted._mil_program.functions['main'].operations))
        actual = converted.predict({'images': sample.numpy()})['result']
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_allclose(actual, torch.nn.functional.silu(sample).numpy(), atol=0.002, rtol=0)

    def test_rank_four_gla_matches_original_recurrence(self):
        from engine.gtr.backbone.gla_portable import portable_gla
        from tools.coreml_gla import coreml_gla, parallel_coreml_gla
        torch.manual_seed(13)
        for length in (35, 1600):
            q, k, v = [torch.randn(2, length, 3, 8) for _ in range(3)]
            gates = -torch.rand_like(q) * 10
            initial = torch.randn(2, 3, 8, 8)
            a, sa = portable_gla(q, k, v, gates, initial_state=initial, output_final_state=True)
            for implementation in (coreml_gla, parallel_coreml_gla):
                b, sb = implementation(q, k, v, gates, initial_state=initial, output_final_state=True)
                torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
                torch.testing.assert_close(sa, sb, atol=2e-5, rtol=2e-5)

        class Attention(torch.nn.Module):
            def forward(self, x):
                return coreml_gla(x, x, x, -x.abs())[0]
        sample = torch.randn(1, 35, 2, 8)
        model = Attention()
        converted = self.convert(model, sample)
        for op in converted._mil_program.functions['main'].operations:
            for out in op.outputs:
                self.assertLessEqual(out.rank, 4)
        with torch.no_grad():
            expected = model(sample).numpy()
        actual = converted.predict({'images': sample.numpy()})['result']
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)

        class ParallelAttention(torch.nn.Module):
            def forward(self, x):
                return parallel_coreml_gla(x, x, x, -x.abs())[0]
        model = ParallelAttention()
        converted = self.convert(model, sample)
        for op in converted._mil_program.functions['main'].operations:
            for out in op.outputs:
                self.assertLessEqual(out.rank, 4)
        with torch.no_grad():
            expected = model(sample).numpy()
        actual = converted.predict({'images': sample.numpy()})['result']
        np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)

    def test_gla_strong_negative_gates_and_chunk_boundary(self):
        import coremltools as ct
        from engine.gtr.backbone.vit_adapter import GatedLinearAttention
        from tools.export_det_coreml import precision_selector

        class Attention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.attention = GatedLinearAttention(hidden_size=16, num_heads=2)

            def forward(self, x):
                return self.attention(x)[0]

        torch.manual_seed(7)
        model = Attention().eval()
        with torch.no_grad():
            model.attention.gk_proj[1].weight.zero_()
            model.attention.gk_proj[1].bias.copy_(torch.linspace(-100, 10, 8))
        model.attention.convert_to_deploy()
        sample = torch.randn(1, 35, 16)
        converted = self.convert(model, sample, ct.transform.FP16ComputePrecision(op_selector=precision_selector))
        for value in (sample, sample * 2):
            with torch.no_grad():
                expected = model(value).numpy()
            actual = converted.predict({'images': value.numpy()})['result']
            self.assertTrue(np.isfinite(actual).all())
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)

    def test_mixed_precision_preserves_value_projection_fp16(self):
        import coremltools as ct
        from engine.gtr.backbone.vit_adapter_spatial_swiglu import SpatialSwiGLU
        from tools.export_det_coreml import precision_selector

        class MLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.mlp = SpatialSwiGLU(hidden_size=16, intermediate_size=32)

            def forward(self, x):
                return self.mlp(x)

        torch.manual_seed(3)
        model = MLP().eval()
        model.mlp.convert_to_deploy()
        sample = torch.randn(1, 16, 16)
        converted = self.convert(model, sample, ct.transform.FP16ComputePrecision(op_selector=precision_selector))
        linears = [op for op in converted._mil_program.functions['main'].operations
                   if op.op_type == 'linear']
        half = [op for op in linears if op.outputs[0].dtype.__name__ == 'fp16']
        self.assertEqual(len(half), 1)
        self.assertIn('value_w', half[0].weight.name)
        self.assertEqual(len(linears), 3)
        with torch.no_grad():
            expected = model(sample).numpy()
        actual = converted.predict({'images': sample.numpy()})['result']
        np.testing.assert_allclose(actual, expected, atol=5e-4, rtol=0)

    def test_numpy_postprocessor_matches_torch(self):
        from engine.gtr.det.postprocessor import PostProcessor
        from tools.inference.coreml_inf import CoreMLDetector
        # Exercise the actual decoder independently of CoreML model loading.
        detector = CoreMLDetector.__new__(CoreMLDetector)
        detector.threshold = 0.05
        detector.num_classes = 366
        detector.num_top_queries = 300
        detector.category_ids = None
        torch.manual_seed(11)
        outputs = {'pred_logits': torch.randn(1, 300, 366) - 5,
                   'pred_boxes': torch.rand(1, 300, 4)}
        post = PostProcessor(num_classes=366, num_top_queries=300).deploy()
        labels, boxes, scores = post(outputs, torch.tensor([[1920, 1080]]))
        keep = scores[0].numpy() > detector.threshold
        actual = detector.postprocess({k: v.numpy() for k, v in outputs.items()}, (1920, 1080))
        np.testing.assert_array_equal(actual['labels'], labels[0].numpy()[keep])
        np.testing.assert_allclose(actual['scores'], scores[0].numpy()[keep], atol=1e-6, rtol=0)
        np.testing.assert_allclose(actual['boxes'], boxes[0].numpy()[keep], atol=1e-4, rtol=0)

    def test_repeated_predictions_keep_input_alive(self):
        import coremltools as ct
        from tools.inference.coreml_inf import CoreMLDetector

        class Outputs(torch.nn.Module):
            def forward(self, x):
                value = x.mean().reshape(1, 1)
                return value.expand(1, 300 * 366), value.expand(1, 1200)

        sample = torch.zeros(1, 3, 4, 4)
        with torch.no_grad():
            traced = torch.jit.trace(Outputs(), sample)
        converted = ct.convert(
            traced, inputs=[ct.TensorType(name='images', shape=sample.shape, dtype=np.float32)],
            outputs=[ct.TensorType(name='pred_logits', dtype=np.float32),
                     ct.TensorType(name='pred_boxes', dtype=np.float32)],
            minimum_deployment_target=ct.target.macOS15,
            compute_precision=ct.precision.FLOAT32, skip_model_load=True,
        )
        converted.user_defined_metadata.update({'input_size': '[4,4]', 'num_classes': '366',
                                                'num_top_queries': '300'})
        with tempfile.TemporaryDirectory() as directory:
            path = directory + '/model.mlpackage'
            converted.save(path)
            with CoreMLDetector(path, compute_units='cpu_only') as detector:
                first = detector.predict_raw(np.full((1, 3, 4, 4), 2, np.float32))
                # Exercise CoreML's delayed reset after the temporary input dies.
                time.sleep(1.3)
                second = detector.predict_raw(np.full((1, 3, 4, 4), 3, np.float32))
                np.testing.assert_array_equal(first['pred_logits'], 2)
                np.testing.assert_array_equal(second['pred_logits'], 3)
            with self.assertRaisesRegex(RuntimeError, 'closed'):
                detector.predict_raw(sample.numpy())

    def test_split_fp16_feature_boundary_and_repeated_predictions(self):
        import coremltools as ct
        from tools.inference.coreml_split_inf import CoreMLSplitDetector
        class Features(torch.nn.Module):
            def forward(self, x):
                value = x.mean().reshape(1, 1).expand(1, 3)
                return value, value + 1, value + 2
        class Decoder(torch.nn.Module):
            def forward(self, a, b, c):
                value = (a.float().mean() + b.float().mean() + c.float().mean()).reshape(1, 1)
                return value.expand(1, 300 * 366), value.expand(1, 1200)
        with torch.no_grad():
            features_trace = torch.jit.trace(Features(), torch.zeros(1, 3, 4, 4))
            decoder_trace = torch.jit.trace(Decoder(), tuple(torch.zeros(1, 3) for _ in range(3)))
        features = ct.convert(features_trace, minimum_deployment_target=ct.target.macOS15,
                              inputs=[ct.TensorType(name='images', shape=(1, 3, 4, 4), dtype=np.float32)],
                              outputs=[ct.TensorType(name=f'level{i}', dtype=np.float16) for i in range(3)],
                              compute_precision=ct.precision.FLOAT16, skip_model_load=True)
        decoder = ct.convert(decoder_trace, minimum_deployment_target=ct.target.macOS15,
                             inputs=[ct.TensorType(name=f'level{i}', shape=(1, 3), dtype=np.float16)
                                     for i in range(3)],
                             outputs=[ct.TensorType(name='pred_logits', dtype=np.float32),
                                      ct.TensorType(name='pred_boxes', dtype=np.float32)],
                             compute_precision=ct.precision.FLOAT32, skip_model_load=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            features.save(str(path / 'features.mlpackage'))
            decoder.save(str(path / 'decoder.mlpackage'))
            manifest = path / 'model.json'
            manifest.write_text(json.dumps({'input_size': [4, 4], 'num_classes': 366,
                'num_top_queries': 300, 'feature_shapes': [[1, 3]] * 3,
                'features_model': 'features.mlpackage', 'decoder_model': 'decoder.mlpackage'}))
            with CoreMLSplitDetector(manifest, compute_units='cpu_only') as detector:
                first = detector.predict_raw(np.full((1, 3, 4, 4), 2, np.float32))
                second = detector.predict_raw(np.full((1, 3, 4, 4), 3, np.float32))
                np.testing.assert_array_equal(first['pred_logits'], 9)
                np.testing.assert_array_equal(second['pred_logits'], 12)
                self.assertTrue(all(x.dtype == np.float32 for x in detector._feature_buffers.values()))
                self.assertTrue(all(out.type.multiArrayType.dataType ==
                                    ct.proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT16
                                    for out in detector.model.get_spec().description.output))
            with self.assertRaisesRegex(RuntimeError, 'closed'):
                detector.predict_raw(np.zeros((1, 3, 4, 4), np.float32))


if __name__ == '__main__':
    unittest.main()
