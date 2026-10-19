"""Export a fixed 640x640 GTR detector to CoreML and verify PyTorch parity.

Run from the repository root:
    uv pip install --python .venv/bin/python -r requirements-coreml.txt
    .venv/bin/python tools/export_det_coreml.py

The checkpoint's EMA weights are used by default, matching torch_inf.py.
The ML Program has float32 normalized NCHW input and raw float32 outputs;
FP16 compute is selected internally, with sensitive operations kept in FP32.
Numerical parity means agreement within the explicit tolerances in the report,
not bitwise equality. Verification requires macOS and runs real CoreML predict.
"""

import argparse
import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import coremltools as ct
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn

from engine.core import YAMLConfig
from tools.inference.coreml_inf import COMPUTE_UNITS, CoreMLDetector

_CONVERSION_OPS_REGISTERED = False
_ANE_LAYOUT = False
_SILU_EXP = False

def register_conversion_ops():
    """Lower supported PyTorch semantics missing from coremltools 9.0.

    Registration is local to this export process. The model's eager forward and
    the PyTorch reference are unchanged. Stable log-sigmoid avoids log(sigmoid)
    underflow; clamp handles a Python integer bound on a floating tensor;
    Integral's vector F.linear weight needs a temporary singleton output axis.
    """
    global _CONVERSION_OPS_REGISTERED
    if _CONVERSION_OPS_REGISTERED:
        return
    from coremltools.converters.mil.frontend.torch.ops import _get_inputs, linear as original_linear
    from coremltools.converters.mil.frontend.torch.torch_op_registry import register_torch_op
    from coremltools.converters.mil.mil import Builder as mb, types

    @register_torch_op
    def log_sigmoid(context, node):
        x = _get_inputs(context, node, expected=1)[0]
        exp = mb.exp(x=mb.mul(x=mb.abs(x=x), y=-1.0))
        result = mb.sub(x=mb.minimum(x=x, y=0.0),
                        y=mb.log(x=mb.add(x=1.0, y=exp)), name=node.name)
        context.add(result)

    @register_torch_op(override=True)
    def clamp_max(context, node):
        x, bound = _get_inputs(context, node, expected=2)
        if x.dtype != bound.dtype:
            bound = mb.cast(x=bound, dtype=types.builtin_to_string(x.dtype))
        context.add(mb.minimum(x=x, y=bound, name=node.name))

    @register_torch_op(override=True)
    def silu(context, node):
        x = _get_inputs(context, node, expected=1)[0]
        if not _SILU_EXP:
            context.add(mb.silu(x=x, name=node.name))
            return
        # ANE's native SiLU approximation showed ~0.015 activation error on
        # this checkpoint. This bounded exponential form avoids both that
        # approximation and exp(-x) overflow for strongly negative inputs.
        exponential = mb.exp(x=mb.mul(x=mb.abs(x=x), y=-1.0))
        inverse = mb.real_div(x=1.0, y=mb.add(x=exponential, y=1.0))
        logistic = mb.select(cond=mb.greater_equal(x=x, y=0.0),
                             a=inverse, b=mb.mul(x=exponential, y=inverse))
        context.add(mb.mul(x=x, y=logistic, name=node.name))

    @register_torch_op(override=True)
    def linear(context, node):
        inputs = _get_inputs(context, node, expected=(2, 3))
        x, weight = inputs[:2]
        if weight.rank != 1:
            if _ANE_LAYOUT and weight.rank == 2 and x.rank == 3:
                bias = inputs[2] if len(inputs) == 3 else None
                channels_first = mb.expand_dims(x=mb.transpose(x=x, perm=[0, 2, 1]), axes=[2])
                kernel = mb.expand_dims(x=weight, axes=[-1, -2])
                out = mb.conv(x=channels_first, weight=kernel, bias=bias)
                context.add(mb.transpose(x=mb.squeeze(x=out, axes=[2]), perm=[0, 2, 1], name=node.name))
                return
            return original_linear(context, node)
        bias = inputs[2] if len(inputs) == 3 else None
        if bias is not None:
            bias = mb.reshape(x=bias, shape=[1])
        out = mb.linear(x=x, weight=mb.expand_dims(x=weight, axes=[0]), bias=bias)
        context.add(mb.squeeze(x=out, axes=[-1], name=node.name))
    _CONVERSION_OPS_REGISTERED = True


class DetectionCoreMLModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        outputs = self.model(images)
        # Flat outputs force a contiguous CoreML boundary. On macOS 26.4 the
        # CPU runtime/binding returns incorrect class rows for a padded rank-3
        # FP32 classifier output (366 channels padded to 368 internally).
        return outputs['pred_logits'].reshape(1, -1), outputs['pred_boxes'].reshape(1, -1)


def load_model(config_path, checkpoint_path, weights='ema'):
    cfg = YAMLConfig(str(config_path))
    if cfg.yaml_cfg.get('task') != 'detection':
        raise ValueError('Expected task: detection')
    backbone = cfg.yaml_cfg[cfg.yaml_cfg['model']]['backbone']
    cfg.yaml_cfg[backbone]['skip_weights_warning'] = True
    model = cfg.model.cpu().eval()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if weights == 'ema' and isinstance(checkpoint.get('ema'), dict):
        state, source = checkpoint['ema']['module'], 'ema.module'
    else:
        state, source = checkpoint.get('model', checkpoint), 'model'
    model.load_state_dict(state, strict=True)
    if model.decoder.num_queries != 300 or not cfg.postprocessor.use_focal_loss:
        raise ValueError('Only 300-query focal-loss detection configs are supported')
    return cfg, model, source


def precision_selector(op):
    """FP16 for value projections in blocks 9-11; FP32 for other math.

    TorchScript scope information survives MIL passes. Keep the full GLA
    module (including its projections), encoder, decoder, convolutions, norms
    and nonlinearities in FP32. This protects cumulative gates, recurrent state, sampling
    coordinates, query ranking, distributional regression, and output heads.
    """
    from coremltools.converters.mil.mil.scope import ScopeSource
    scope_types = op.scopes.get(ScopeSource.TORCHSCRIPT_MODULE_TYPE, [])
    # Early projection rounding propagates into low-confidence query selection.
    # Limit FP16 to late value projections to satisfy the raw-output check.
    # Deploy splits the original gate_proj into named gate_w/value_w tensors.
    if 'SpatialSwiGLU' not in scope_types or op.op_type != 'linear' or 'value_w' not in op.weight.name:
        return False
    block = re.search(r'blocks[._](\d+)[._]mlp', op.weight.name)
    # Standalone SpatialSwiGLU modules have no block prefix (used by regression tests).
    return block is None or int(block[1]) in {9, 10, 11}


def precision_statistics(program):
    counts = Counter()
    def visit(block):
        for op in block.operations:
            for child in op.blocks:
                visit(child)
            for out in op.outputs:
                dtype = str(out.dtype.__name__)
                if dtype in {'fp16', 'fp32'} and op.op_type != 'const':
                    counts[dtype] += 1
    for function in program.functions.values():
        visit(function)
    return dict(counts)


def fp16_except_norms(op):
    """Keep normalization in FP32 while projections and GLA stay FP16."""
    from coremltools.converters.mil.mil.scope import ScopeSource
    scopes = op.scopes.get(ScopeSource.TORCHSCRIPT_MODULE_TYPE, [])
    return op.op_type != 'layer_norm' and not any(
        'LayerNorm' in scope or 'RMSNorm' in scope for scope in scopes)


def trace_model(model, cfg, ane_layout=False, gla_mode='sequential', class_padding=0):
    deployed = copy.deepcopy(model).deploy().eval()
    if class_padding:
        original_classes = deployed.decoder.num_classes
        padded_classes = ((original_classes + class_padding - 1) // class_padding) * class_padding
        for module in deployed.decoder.modules():
            if isinstance(module, nn.Linear) and module.out_features == original_classes:
                with torch.no_grad():
                    module.weight = nn.Parameter(torch.cat((module.weight,
                        module.weight.new_zeros(padded_classes - original_classes, module.in_features))))
                    module.bias = nn.Parameter(torch.cat((module.bias,
                        module.bias.new_full((padded_classes - original_classes,), -10000))))
                module.out_features = padded_classes
        deployed.decoder.num_classes = padded_classes
    # Deploy's atlas optimization relies on in-place writes and enlarges sampling
    # coordinates. Use the equivalent per-level grid_sample path for CoreML.
    deployed.decoder.decoder._va_ready = False
    wrapper = DetectionCoreMLModel(deployed).eval()
    size = cfg.yaml_cfg['eval_spatial_size']
    sample = torch.zeros(1, 3, *size)
    fastpath = torch.backends.mha.get_fastpath_enabled()
    from engine.gtr.backbone import vit_adapter
    original_gla = vit_adapter.portable_gla
    try:
        if ane_layout:
            from tools.coreml_gla import coreml_gla, parallel_coreml_gla
            if gla_mode == 'parallel_direct':
                from functools import partial
                vit_adapter.portable_gla = partial(parallel_coreml_gla, direct_gate_sums=True)
            else:
                vit_adapter.portable_gla = parallel_coreml_gla if gla_mode == 'parallel' else coreml_gla
        torch.backends.mha.set_fastpath_enabled(False)
        with torch.inference_mode():
            traced = torch.jit.trace(wrapper, sample, check_trace=False)
    finally:
        vit_adapter.portable_gla = original_gla
        torch.backends.mha.set_fastpath_enabled(fastpath)
    return wrapper, traced, sample


def export(model, cfg, output_path, metadata, precision='mixed', ane_layout=False, gla_mode='sequential', silu_mode='native', class_padding=0):
    global _ANE_LAYOUT, _SILU_EXP
    _ANE_LAYOUT = ane_layout
    _SILU_EXP = silu_mode == 'exp'
    register_conversion_ops()
    wrapper, traced, sample = trace_model(model, cfg, ane_layout, gla_mode, class_padding)
    compute_precision = (ct.precision.FLOAT16 if precision == 'fp16' else
                         ct.transform.FP16ComputePrecision(op_selector=(
                             fp16_except_norms if precision == 'fp16_norms_fp32' else precision_selector)))
    mlmodel = ct.convert(
        traced, convert_to='mlprogram', minimum_deployment_target=ct.target.macOS15,
        inputs=[ct.TensorType(name='images', shape=sample.shape, dtype=np.float32)],
        outputs=[ct.TensorType(name='pred_logits', dtype=np.float32),
                 ct.TensorType(name='pred_boxes', dtype=np.float32)],
        compute_precision=compute_precision, compute_units=ct.ComputeUnit.CPU_AND_GPU,
        skip_model_load=True,
    )
    statistics = precision_statistics(mlmodel._mil_program)
    if not statistics.get('fp16'):
        raise RuntimeError('Export unexpectedly contains no FP16 operations')
    metadata['precision'] = precision
    metadata['ane_layout'] = ane_layout
    metadata['gla_mode'] = gla_mode if ane_layout else 'original'
    metadata['silu_mode'] = silu_mode
    metadata['class_padding'] = class_padding
    metadata['output_class_slots'] = wrapper.model.decoder.num_classes
    if precision == 'fp16':
        metadata['fp16_policy'] = 'All eligible floating operations FP16; float32 input/output boundaries'
    elif precision == 'fp16_norms_fp32':
        metadata['fp16_policy'] = 'FP16 except LayerNorm and RMSNorm modules FP32; float32 input/output boundaries'
    metadata['precision_statistics'] = statistics
    for key, value in metadata.items():
        mlmodel.user_defined_metadata[key] = json.dumps(value, ensure_ascii=False)
    mlmodel.short_description = 'GTR Objects365 detector, FP16 with selective FP32 compute'
    mlmodel.input_description['images'] = 'Normalized RGB NCHW float32, batch 1; ImageNet mean/std'
    mlmodel.output_description['pred_logits'] = f'Raw class logits flattened to [1,{300 * wrapper.model.decoder.num_classes}]'
    mlmodel.output_description['pred_boxes'] = 'Normalized cxcywh boxes flattened to [1,1200]'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output_path))
    return wrapper, traced, statistics


def sigmoid(logits):
    # Diagnostics in float64 avoid overflow for very negative background logits.
    values = logits.astype(np.float64)
    return np.exp(-np.logaddexp(0, -values))


def align_queries(expected, actual):
    """Queries are an unordered set: top-k may permute near-equal seed scores.

    Match all 300 queries using boxes AND all class probabilities; report the
    original ordered error separately so permutations cannot hide from readers.
    """
    reference_boxes, actual_boxes = expected['pred_boxes'][0], actual['pred_boxes'][0]
    reference_scores, actual_scores = sigmoid(expected['pred_logits'][0]), sigmoid(actual['pred_logits'][0])
    costs = np.abs(reference_boxes[:, None] - actual_boxes[None]).sum(-1)
    # Avoid a [300,300,366] allocation.
    for category in range(reference_scores.shape[-1]):
        costs += np.abs(reference_scores[:, category, None] - actual_scores[None, :, category])
    rows, columns = linear_sum_assignment(costs)
    return rows, columns, reference_scores, actual_scores


def verification_samples(image_paths, detector, synthetic=True):
    for path in image_paths:
        image = np.asarray(Image.open(path).convert('RGB'))[:, :, ::-1].copy()
        yield str(path), image, detector.preprocess(image)
        yield str(path) + ':horizontal_flip', image[:, ::-1].copy(), detector.preprocess(image[:, ::-1])
    if synthetic:
        height, width = detector.input_size
        rng = np.random.default_rng(7)
        for name, image in (
            ('black', np.zeros((height, width, 3), np.uint8)),
            ('white', np.full((height, width, 3), 255, np.uint8)),
            ('random', rng.integers(0, 256, (height, width, 3), dtype=np.uint8)),
        ):
            yield name, image, detector.preprocess(image)


def verify(model, wrapper, traced, cfg, output_path, image_paths, args, detector_class=CoreMLDetector):
    detector = detector_class(output_path, args.threshold, args.compute_units)
    postprocessor = cfg.postprocessor.deploy()
    results, failures = [], []
    fastpath = torch.backends.mha.get_fastpath_enabled()
    try:
        torch.backends.mha.set_fastpath_enabled(False)
        for name, image, sample in verification_samples(image_paths, detector):
            print(f'Verifying {name} ({args.compute_units})', flush=True)
            tensor = torch.from_numpy(sample)
            with torch.inference_mode():
                reference = model(tensor)
                expected = {key: reference[key].numpy() for key in ('pred_logits', 'pred_boxes')}
                deploy_out = wrapper(tensor)
                traced_out = traced(tensor)
                sizes = torch.tensor([[image.shape[1], image.shape[0]]])
                torch_labels, torch_boxes, torch_scores = postprocessor(reference, sizes)
            actual = detector.predict_raw(sample)
            row, column, reference_probs, actual_probs = align_queries(expected, actual)
            entry = {'sample': name, 'ordered_raw_max_abs_error': {},
                     'aligned_raw_max_abs_error': {}, 'deploy_max_abs_error': {}}
            deployed_outputs = {key: (deploy_out[index].numpy().reshape(1, 300, -1)[..., :expected[key].shape[-1]])
                                for index, key in enumerate(('pred_logits', 'pred_boxes'))}
            traced_outputs = {key: (traced_out[index].numpy().reshape(1, 300, -1)[..., :expected[key].shape[-1]])
                              for index, key in enumerate(('pred_logits', 'pred_boxes'))}
            trace_row, trace_column, trace_reference_probs, trace_actual_probs = align_queries(
                deployed_outputs, traced_outputs)
            entry['trace_aligned_probability_max_abs_error'] = float(np.max(np.abs(
                trace_reference_probs[trace_row] - trace_actual_probs[trace_column])))
            entry['trace_aligned_box_max_abs_error_normalized'] = float(np.max(np.abs(
                deployed_outputs['pred_boxes'][0, trace_row] - traced_outputs['pred_boxes'][0, trace_column])))
            if args.parity == 'raw' and (entry['trace_aligned_probability_max_abs_error'] > 1e-4 or
                                        entry['trace_aligned_box_max_abs_error_normalized'] > 2e-5):
                failures.append(f'{name}: Torch trace raw outputs differ from deploy')
            deploy_row, deploy_column, deploy_reference_probs, deploy_actual_probs = align_queries(
                expected, deployed_outputs)
            entry['deploy_permuted_queries'] = int(np.count_nonzero(deploy_row != deploy_column))
            entry['deploy_aligned_probability_max_abs_error'] = float(np.max(np.abs(
                deploy_reference_probs[deploy_row] - deploy_actual_probs[deploy_column])))
            entry['deploy_aligned_box_max_abs_error_normalized'] = float(np.max(np.abs(
                expected['pred_boxes'][0, deploy_row] - deployed_outputs['pred_boxes'][0, deploy_column])))
            # Deploy folding can also permute near-equal query seeds. Record
            # unmatched differences, but check the same unordered semantics as
            # CoreML. Keep reporting all samples if export equivalence fails.
            if args.parity == 'raw' and entry['deploy_aligned_probability_max_abs_error'] > 1e-4:
                failures.append(f'{name}: Torch deploy class probabilities exceed 0.0001')
            if args.parity == 'raw' and entry['deploy_aligned_box_max_abs_error_normalized'] > 2e-5:
                failures.append(f'{name}: Torch deploy boxes exceed 0.00002')
            for index, key in enumerate(('pred_logits', 'pred_boxes')):
                if not np.isfinite(expected[key]).all():
                    raise AssertionError(f'Non-finite PyTorch {key}: {name}')
                # Export-only changes must agree before measuring FP16 rounding.
                deployed = deployed_outputs[key]
                entry['deploy_max_abs_error'][key] = float(np.max(np.abs(deployed - expected[key])))
                if not np.isfinite(traced_outputs[key]).all():
                    raise AssertionError(f'Non-finite Torch trace {key}: {name}')
                entry['ordered_raw_max_abs_error'][key] = float(np.max(np.abs(expected[key] - actual[key])))
                entry['aligned_raw_max_abs_error'][key] = float(np.max(np.abs(expected[key][0, row] - actual[key][0, column])))
            entry['permuted_queries'] = int(np.count_nonzero(row != column))
            entry['raw_probability_max_abs_error'] = float(np.max(np.abs(reference_probs[row] - actual_probs[column])))
            entry['raw_box_max_abs_error_normalized'] = entry['aligned_raw_max_abs_error']['pred_boxes']
            if args.parity == 'raw' and entry['raw_probability_max_abs_error'] > args.raw_score_atol:
                failures.append(f'{name}: raw class probabilities exceed {args.raw_score_atol}')
            if args.parity == 'raw' and entry['raw_box_max_abs_error_normalized'] > args.raw_box_atol:
                failures.append(f'{name}: raw boxes exceed {args.raw_box_atol}')
            decoded = detector.postprocess(actual, (image.shape[1], image.shape[0]))
            keep = torch_scores[0].numpy() > args.threshold
            labels = torch_labels[0].numpy()[keep]
            if detector.category_ids is not None:
                labels = np.asarray(detector.category_ids)[labels]
            boxes = torch_boxes[0].numpy()[keep]
            scores = torch_scores[0].numpy()[keep]
            entry['torch_detections'], entry['coreml_detections'] = len(labels), len(decoded['labels'])
            entry['labels_exact'] = len(labels) == len(decoded['labels'])
            entry['detection_score_max_abs_error'] = 0.0
            entry['detection_box_max_abs_error_px'] = 0.0
            if not entry['labels_exact']:
                failures.append(f'{name}: detection counts differ')
            elif len(labels):
                costs = np.abs(boxes[:, None] - decoded['boxes'][None]).sum(-1)
                costs += (labels[:, None] != decoded['labels'][None]) * 1e9
                rows, columns = linear_sum_assignment(costs)
                entry['labels_exact'] = bool(np.array_equal(labels[rows], decoded['labels'][columns]))
                entry['detection_score_max_abs_error'] = float(np.max(np.abs(scores[rows] - decoded['scores'][columns])))
                entry['detection_box_max_abs_error_px'] = float(np.max(np.abs(boxes[rows] - decoded['boxes'][columns])))
                if not entry['labels_exact']:
                    failures.append(f'{name}: detection classes differ')
                if entry['detection_score_max_abs_error'] > args.score_atol:
                    failures.append(f'{name}: detection scores exceed {args.score_atol}')
                if entry['detection_box_max_abs_error_px'] > args.box_atol_px:
                    failures.append(f'{name}: detection boxes exceed {args.box_atol_px}px')
            # Independently check the numpy decoder against the real Torch postprocessor.
            decoded_reference = detector.postprocess(expected, (image.shape[1], image.shape[0]))
            np.testing.assert_array_equal(decoded_reference['labels'], labels)
            np.testing.assert_allclose(decoded_reference['scores'], scores, atol=1e-6, rtol=0)
            np.testing.assert_allclose(decoded_reference['boxes'], boxes, atol=1e-4, rtol=0)
            results.append(entry)
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)
        detector.close()
    return {
        'passed': not failures, 'failures': failures, 'compute_units': args.compute_units,
        'torch_device': 'cpu', 'torch_dtype': 'float32', 'detection_threshold': args.threshold,
        'parity_mode': args.parity,
        'decoder_compute_units': getattr(detector, 'decoder_compute_units', None),
        'tolerances': {'raw_probability_abs': args.raw_score_atol,
                       'raw_box_abs_normalized': args.raw_box_atol,
                       'detection_score_abs': args.score_atol, 'detection_box_abs_px': args.box_atol_px},
        'query_matching': 'Hungarian matching on boxes and every class probability; detection class must match exactly',
        'samples': results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/det/obj365_pretrain/gtr_s.yml'))
    parser.add_argument('--checkpoint', type=Path, default=Path('weights/gtr_s_obj365.pth'))
    parser.add_argument('--output', type=Path, default=Path('weights/gtr_s_obj365_fp16.mlpackage'))
    parser.add_argument('--weights', choices=['ema', 'model'], default='ema')
    parser.add_argument('--precision', choices=['mixed', 'fp16', 'fp16_norms_fp32'], default='mixed')
    parser.add_argument('--parity', choices=['raw', 'detections'], default='raw',
                        help='Require all raw queries or only detections above threshold; always report raw differences')
    parser.add_argument('--ane-layout', action='store_true',
                        help='Experimental rank-4 GLA and 1x1 convolution projection lowering (FP16 only)')
    parser.add_argument('--gla-mode', choices=['sequential', 'parallel', 'parallel_direct'], default='sequential',
                        help='Parallel uses a batched associative chunk scan; requires ane-layout')
    parser.add_argument('--silu-mode', choices=['native', 'exp'], default='native',
                        help='Bounded exp decomposition avoids the native ANE SiLU approximation')
    parser.add_argument('--class-padding', type=int, choices=[0, 32], default=0,
                        help='Pad classifier channels with inactive dummy classes for aligned CoreML buffers')
    parser.add_argument('--verify-images', type=Path, nargs='+', default=[Path('assets/00.jpg')])
    parser.add_argument('--compute-units', choices=COMPUTE_UNITS, default='cpu_and_gpu')
    parser.add_argument('--threshold', type=float, default=0.4)
    parser.add_argument('--score-atol', type=float, default=0.002)
    parser.add_argument('--box-atol-px', type=float, default=0.5)
    parser.add_argument('--raw-score-atol', type=float, default=0.005)
    parser.add_argument('--raw-box-atol', type=float, default=0.005)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--verify-only', action='store_true', help='Verify an existing package without converting again')
    parser.add_argument('--report', type=Path, help='Optional verification report path')
    args = parser.parse_args()
    if args.ane_layout and args.precision not in {'fp16', 'fp16_norms_fp32'} and not args.verify_only:
        parser.error('ane-layout requires precision fp16')
    if args.gla_mode != 'sequential' and not args.ane_layout and not args.verify_only:
        parser.error('parallel gla-mode requires ane-layout')
    if args.output.suffix != '.mlpackage':
        parser.error('--output must end with .mlpackage')
    if not 0 <= args.threshold <= 1 or min(args.score_atol, args.box_atol_px, args.raw_score_atol, args.raw_box_atol) < 0:
        parser.error('Invalid threshold or tolerance')
    images = []
    for path in args.verify_images:
        if path.is_dir():
            images.extend(sorted(p for p in path.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}))
        elif path.is_file():
            images.append(path)
        else:
            parser.error(f'Image path does not exist: {path}')
    if not images:
        parser.error('At least one real verification image is required')
    torch.set_num_threads(args.threads)
    cfg, model, source = load_model(args.config, args.checkpoint, args.weights)
    metadata = {
        'input_size': cfg.yaml_cfg['eval_spatial_size'], 'num_classes': cfg.postprocessor.num_classes,
        'num_top_queries': cfg.postprocessor.num_top_queries, 'weight_source': source,
        'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        'torch_version': torch.__version__, 'coremltools_version': ct.__version__,
        'output_layout': 'flat',
        'fp16_policy': 'SpatialSwiGLU value projections in blocks 9-11; all other operations FP32',
    }
    if cfg.postprocessor.remap_mscoco_category:
        from engine.data.dataset.coco_dataset import mscoco_label2category
        metadata['category_ids'] = [mscoco_label2category[index] for index in range(cfg.postprocessor.num_classes)]
    if args.verify_only:
        saved_model = ct.models.MLModel(str(args.output), skip_model_load=True)
        saved_metadata = saved_model.user_defined_metadata
        for key in ('checkpoint_sha256', 'weight_source', 'num_classes', 'input_size', 'output_layout'):
            if json.loads(saved_metadata[key]) != metadata[key]:
                raise ValueError(f'Existing package does not match {key}')
        metadata = {key: json.loads(value) for key, value in saved_metadata.items()
                    if not key.startswith('com.github.apple.')}
        wrapper, traced, _ = trace_model(model, cfg, metadata.get('ane_layout', False), metadata.get('gla_mode', 'sequential'), metadata.get('class_padding', 0))
        statistics = metadata['precision_statistics']
    else:
        print(f'Exporting {args.checkpoint} ({source}, {args.precision})', flush=True)
        wrapper, traced, statistics = export(model, cfg, args.output, metadata, args.precision, args.ane_layout, args.gla_mode, args.silu_mode, args.class_padding)
    report = {'model_path': str(args.output), 'config': str(args.config), 'metadata': metadata,
              'precision_statistics': statistics,
              'verification': verify(model, wrapper, traced, cfg, args.output, images, args)}
    report_path = args.report or args.output.with_suffix('.verification.json')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(f'Saved {args.output} and {report_path}', flush=True)
    if not report['verification']['passed']:
        raise SystemExit('CoreML parity failed: ' + '; '.join(report['verification']['failures']))
    print('CoreML parity passed', flush=True)


if __name__ == '__main__':
    main()
