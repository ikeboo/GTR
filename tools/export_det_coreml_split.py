"""Export GTR as FP16 CPU/ANE features plus an FP32 CPU/GPU decoder.

The split keeps the heavy feature extraction eligible for ANE execution while
avoiding the large detection differences seen with a CPU/ANE-only decoder on
macOS 26.4. Native ANE SiLU is replaced with a bounded exp decomposition.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import coremltools as ct
import numpy as np
import torch
from torch import nn
import tools.export_det_coreml as common
from tools.coreml_gla import parallel_coreml_gla
from tools.inference.coreml_split_inf import CoreMLSplitDetector


class FeatureModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone, self.encoder = model.backbone, model.encoder

    def forward(self, images):
        return tuple(value.reshape(1, -1) for value in self.encoder(self.backbone(images)))


class DecoderModel(nn.Module):
    def __init__(self, model, feature_shapes):
        super().__init__()
        self.decoder, self.feature_shapes = model.decoder, feature_shapes

    def forward(self, level0, level1, level2):
        outputs = self.decoder([value.reshape(shape) for value, shape in
                               zip((level0, level1, level2), self.feature_shapes)])
        return outputs['pred_logits'].reshape(1, -1), outputs['pred_boxes'].reshape(1, -1)


def export_parts(model, cfg, manifest_path, metadata):
    wrapper, traced, sample = common.trace_model(model, cfg, ane_layout=True, gla_mode='parallel')
    deployed = wrapper.model
    with torch.inference_mode():
        levels = deployed.encoder(deployed.backbone(sample))
    shapes = [tuple(value.shape) for value in levels]
    from engine.gtr.backbone import vit_adapter
    original_gla = vit_adapter.portable_gla
    fastpath = torch.backends.mha.get_fastpath_enabled()
    try:
        torch.backends.mha.set_fastpath_enabled(False)
        vit_adapter.portable_gla = parallel_coreml_gla
        with torch.inference_mode():
            features_trace = torch.jit.trace(FeatureModel(deployed).eval(), sample, check_trace=False)
            decoder_trace = torch.jit.trace(DecoderModel(deployed, shapes).eval(),
                                           tuple(value.reshape(1, -1) for value in levels), check_trace=False)
    finally:
        vit_adapter.portable_gla = original_gla
        torch.backends.mha.set_fastpath_enabled(fastpath)
    common._ANE_LAYOUT = common._SILU_EXP = True
    common.register_conversion_ops()
    features = ct.convert(features_trace, minimum_deployment_target=ct.target.macOS15,
                          inputs=[ct.TensorType(name='images', shape=sample.shape, dtype=np.float32)],
                          outputs=[ct.TensorType(name=f'level{i}', dtype=np.float16) for i in range(3)],
                          compute_precision=ct.precision.FLOAT16, skip_model_load=True)
    decoder = ct.convert(decoder_trace, minimum_deployment_target=ct.target.macOS15,
                         inputs=[ct.TensorType(name=f'level{i}', shape=(1, int(np.prod(shape))), dtype=np.float16)
                                 for i, shape in enumerate(shapes)],
                         outputs=[ct.TensorType(name='pred_logits', dtype=np.float32),
                                  ct.TensorType(name='pred_boxes', dtype=np.float32)],
                         compute_precision=ct.precision.FLOAT32, skip_model_load=True)
    metadata.update(feature_shapes=shapes, features_precision='fp16', decoder_precision='fp32',
                    gla_mode='parallel', silu_mode='exp',
                    precision_statistics={'features': common.precision_statistics(features._mil_program),
                                          'decoder': common.precision_statistics(decoder._mil_program)})
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    for role, artifact in [('features', features), ('decoder', decoder)]:
        filename = f'{manifest_path.stem}_{role}.mlpackage'
        metadata[f'{role}_model'] = filename
        artifact.short_description = f'GTR Objects365 split detector: {role}'
        artifact.user_defined_metadata.update({key: json.dumps(value) for key, value in metadata.items()})
        artifact.save(str(manifest_path.parent / filename))
    manifest_path.write_text(json.dumps(metadata, indent=2) + '\n')
    return wrapper, traced


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/det/obj365_pretrain/gtr_s.yml'))
    parser.add_argument('--checkpoint', type=Path, default=Path('weights/gtr_s_obj365.pth'))
    parser.add_argument('--output', type=Path, default=Path('weights/gtr_s_obj365_ane_split.json'))
    parser.add_argument('--weights', choices=['ema', 'model'], default='ema')
    parser.add_argument('--verify-images', type=Path, nargs='+', default=[Path('assets/00.jpg')])
    parser.add_argument('--compute-units', choices=['cpu_and_ne', 'cpu_and_gpu', 'all'], default='cpu_and_ne')
    parser.add_argument('--threshold', type=float, default=0.4)
    parser.add_argument('--score-atol', type=float, default=0.01)
    parser.add_argument('--box-atol-px', type=float, default=2.0)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if args.output.suffix != '.json':
        parser.error('output must be a .json manifest path')
    if not 0 <= args.threshold <= 1 or min(args.score_atol, args.box_atol_px) < 0:
        parser.error('Invalid threshold or tolerance')
    images = []
    for path in args.verify_images:
        if path.is_dir():
            images.extend(sorted(p for p in path.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}))
        elif path.is_file():
            images.append(path)
        else:
            parser.error(f'Image does not exist: {path}')
    if not images:
        parser.error('At least one verification image is required')
    torch.set_num_threads(args.threads)
    cfg, model, source = common.load_model(args.config, args.checkpoint, args.weights)
    metadata = {'input_size': cfg.yaml_cfg['eval_spatial_size'], 'num_classes': cfg.postprocessor.num_classes,
                'num_top_queries': cfg.postprocessor.num_top_queries, 'weight_source': source,
                'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                'torch_version': torch.__version__, 'coremltools_version': ct.__version__,
                'features_compute_units': args.compute_units, 'decoder_compute_units': 'cpu_and_gpu'}
    if cfg.postprocessor.remap_mscoco_category:
        from engine.data.dataset.coco_dataset import mscoco_label2category
        metadata['category_ids'] = [mscoco_label2category[i] for i in range(cfg.postprocessor.num_classes)]
    if args.verify_only:
        saved = json.loads(args.output.read_text())
        for key in ('checkpoint_sha256', 'weight_source', 'input_size', 'num_classes'):
            if metadata[key] != saved[key]:
                raise ValueError(f'Manifest does not match {key}')
        metadata = saved
        wrapper, traced, _ = common.trace_model(model, cfg, ane_layout=True, gla_mode='parallel')
    else:
        wrapper, traced = export_parts(model, cfg, args.output, metadata)
    args.parity, args.raw_score_atol, args.raw_box_atol = 'detections', 0.005, 0.005
    verification = common.verify(model, wrapper, traced, cfg, args.output, images, args, CoreMLSplitDetector)
    report = {'model_path': str(args.output), 'metadata': metadata, 'verification': verification}
    report_path = args.output.with_suffix('.verification.json')
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Saved {args.output} and {report_path}', flush=True)
    if not verification['passed']:
        raise SystemExit('Detection parity failed: ' + '; '.join(verification['failures']))
    print('Detection parity passed', flush=True)


if __name__ == '__main__':
    main()
