"""Export a GTR pose checkpoint to standard ONNX and check PyTorch parity.

The exported graph has fixed batch and image dimensions and includes the pose
postprocessor. It uses PyTorch GLA operations, so ONNX Runtime needs no GTR plugin.

Usage (from the repository root, with dependencies installed in .venv):
    .venv/bin/python tools/export_pose_onnx.py \
        --config configs/pose/custom_pose_m.yml \
        --checkpoint outputs/custom_pose_m/best.pth \
        --output outputs/custom_pose_m/best.onnx \
        --verify-images assets/00.jpg

--verify-images accepts an image or a directory. When given, those images are used
to compare PyTorch, ONNX Runtime, and repos/inferencers/ec_pose.py inference.
--verify-device selects the PyTorch comparison device (cpu by default, or mps);
ONNX Runtime verification uses CPU in either case. The command also writes
a .verification.json file beside the ONNX model with metadata and check results.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import onnx
import torch
from torch import nn

from engine.core import YAMLConfig
from repos.inferencers.ec_pose import ECPose


class PoseOnnxModel(nn.Module):
    def __init__(self, model, postprocessor):
        super().__init__()
        self.model = model
        self.postprocessor = postprocessor

    def forward(self, images, orig_target_sizes):
        return self.postprocessor(self.model(images), orig_target_sizes)


def load_model(config_path, checkpoint_path):
    cfg = YAMLConfig(str(config_path))
    if cfg.yaml_cfg.get('task') != 'pose':
        raise ValueError('Expected a pose config')
    backbone_name = cfg.yaml_cfg[cfg.yaml_cfg['model']]['backbone']
    cfg.yaml_cfg[backbone_name]['skip_weights_warning'] = True
    model = cfg.model
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    ema = checkpoint.get('ema') if isinstance(checkpoint, dict) else None
    state = ema['module'] if isinstance(ema, dict) and 'module' in ema else checkpoint.get('model', checkpoint)
    model.load_state_dict(state, strict=True)
    wrapper = PoseOnnxModel(model.deploy(), cfg.postprocessor.deploy()).eval()
    return cfg, wrapper


def make_metadata(cfg):
    config = cfg.yaml_cfg
    model_name = config['model']
    decoder_name = config[model_name]['decoder']
    post_name = config['postprocessor']
    metadata = {
        'num_keypoints': config[post_name]['num_body_points'],
        'input_size': config['eval_spatial_size'],
        'remap_categories': bool(config.get('remap_mscoco_category', False)),
        'num_model_classes': config[decoder_name].get('num_classes', config['num_classes']),
    }
    annotation_path = config.get('val_dataloader', {}).get('dataset', {}).get('ann_file')
    if annotation_path and Path(annotation_path).is_file():
        categories = json.loads(Path(annotation_path).read_text())['categories']
        categories = sorted(categories, key=lambda category: category['id'])
        metadata.update({
            'category_ids': [category['id'] for category in categories],
            'category_names': [category['name'] for category in categories],
            'keypoint_names': categories[0].get('keypoints', []) if categories else [],
            'skeleton': categories[0].get('skeleton', []) if categories else [],
        })
    return metadata


def export(wrapper, output_path, image_size, metadata):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = image_size
    example_images = torch.zeros(1, 3, height, width)
    example_sizes = torch.tensor([[width, height]], dtype=torch.int64)
    # The evaluation fast path of MultiheadAttention has no legacy ONNX symbolic.
    # Disable it only while tracing; the parity check uses normal PyTorch eval.
    fastpath = torch.backends.mha.get_fastpath_enabled()
    try:
        torch.backends.mha.set_fastpath_enabled(False)
        with torch.inference_mode():
            torch.onnx.export(
                wrapper, (example_images, example_sizes), str(output_path),
                input_names=['images', 'orig_target_sizes'],
                output_names=['scores', 'labels', 'keypoints'],
                opset_version=18, dynamo=False, do_constant_folding=True,
            )
    finally:
        torch.backends.mha.set_fastpath_enabled(fastpath)

    graph = onnx.load(output_path)
    del graph.metadata_props[:]
    for name, value in metadata.items():
        item = graph.metadata_props.add()
        item.key = name
        item.value = json.dumps(value, ensure_ascii=False)
    onnx.checker.check_model(graph)
    custom_domains = {node.domain for node in graph.graph.node if node.domain}
    if custom_domains:
        raise RuntimeError(f'Export contains custom ONNX operators: {sorted(custom_domains)}')
    onnx.save(graph, output_path)


def verify(wrapper, output_path, image_paths, torch_device='cpu', threshold=0.4):
    if torch_device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS is not available for PyTorch parity checking')
    wrapper = wrapper.to(torch_device)
    inferencer = ECPose(output_path, thresh=threshold, providers=['CPUExecutionProvider'])
    max_score_error = 0.0
    max_keypoint_error = 0.0
    detections = 0
    image_results = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f'Failed to read {image_path}')
        images, original_size = inferencer._preprocess(image)
        with torch.inference_mode():
            reference = wrapper(torch.from_numpy(images).to(torch_device),
                                torch.from_numpy(original_size).to(torch_device))
        expected = [tensor.cpu().numpy() for tensor in reference]
        actual = inferencer.session.run(None, {
            inferencer.image_input_name: images,
            inferencer.size_input_name: original_size,
        })
        for name, reference_output, onnx_output in zip(
                ('scores', 'labels', 'keypoints'), expected, actual):
            if not (np.isfinite(reference_output).all() and np.isfinite(onnx_output).all()):
                raise AssertionError(f'Non-finite {name} on {image_path}')
        if not np.array_equal(expected[1], actual[1]):
            raise AssertionError(f'Label mismatch on {image_path}')
        score_error = float(np.max(np.abs(expected[0] - actual[0])))
        keypoint_error = float(np.max(np.abs(expected[2] - actual[2])))
        max_score_error = max(max_score_error, score_error)
        max_keypoint_error = max(max_keypoint_error, keypoint_error)

        result = inferencer(image)
        valid_labels = np.array([
            inferencer._map_label(int(label)) is not None for label in expected[1][0]
        ])
        keep = (expected[0][0] > threshold) & valid_labels
        expected_count = int(np.count_nonzero(keep))
        if len(result) != expected_count:
            raise AssertionError(f'ECPose detection count mismatch on {image_path}')
        expected_classes = [inferencer._map_label(int(label))[0]
                            for label in expected[1][0][keep]]
        if result.classes != expected_classes:
            raise AssertionError(f'ECPose class mismatch on {image_path}')
        if not np.allclose(result.scores, expected[0][0][keep], atol=1e-4, rtol=0):
            raise AssertionError(f'ECPose score mismatch on {image_path}')
        detections += len(result)
        image_results.append({
            'image_path': str(image_path),
            'detections': len(result),
            'classes': result.classes,
            'scores': result.scores,
            'keypoints': result.kpts,
        })
        if result.kpts:
            expected_xy = expected[2][0][keep].copy()
            expected_xy[..., 0] = np.clip(expected_xy[..., 0], 0, image.shape[1] - 1)
            expected_xy[..., 1] = np.clip(expected_xy[..., 1], 0, image.shape[0] - 1)
            decoded = np.asarray(result.kpts, dtype=np.float32).reshape(expected_xy.shape)
            if not np.allclose(decoded, expected_xy, atol=0.002, rtol=0):
                raise AssertionError(f'ECPose keypoint mismatch on {image_path}')

    if max_score_error > 1e-4 or max_keypoint_error > 0.002:
        raise AssertionError(
            f'ONNX parity failed: score={max_score_error}, keypoint_px={max_keypoint_error}'
        )
    default_inferencer = ECPose(output_path, thresh=threshold)
    first_image = cv2.imread(str(image_paths[0]))
    default_result = default_inferencer(first_image)
    first_result = image_results[0]
    if (default_result.classes != first_result['classes'] or
            not np.allclose(default_result.scores, first_result['scores'], atol=1e-4, rtol=0) or
            not np.allclose(default_result.kpts, first_result['keypoints'], atol=0.002, rtol=0)):
        raise AssertionError('Default ECPose provider differs from CPU inference')
    return {
        'images': len(image_paths),
        'ecpose_detections': detections,
        'detection_threshold': threshold,
        'image_results': image_results,
        'max_score_abs_error': max_score_error,
        'max_keypoint_abs_error_px': max_keypoint_error,
        'score_tolerance': 1e-4,
        'keypoint_tolerance_px': 0.002,
        'labels_exact': True,
        'torch_device': torch_device,
        'execution_provider': inferencer.session.get_providers(),
        'default_execution_provider': default_inferencer.session.get_providers(),
        'default_inferencer_detections': len(default_result),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--verify-images', type=Path,
                        help='Image or directory for PyTorch/ONNX/ECPose comparison')
    parser.add_argument('--verify-device', choices=['cpu', 'mps'], default='cpu',
                        help='PyTorch device for comparison (ONNX Runtime uses CPU)')
    parser.add_argument('--verify-threshold', type=float, default=0.4,
                        help='Detection threshold for ECPose checks (default: 0.4)')
    args = parser.parse_args()
    if not 0 <= args.verify_threshold <= 1:
        parser.error('--verify-threshold must be between 0 and 1')

    image_paths = []
    if args.verify_images:
        if args.verify_images.is_file():
            image_paths = [args.verify_images]
        elif args.verify_images.is_dir():
            image_paths = sorted(path for path in args.verify_images.iterdir()
                                 if path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'})
        if not image_paths:
            parser.error(f'No images in {args.verify_images}')

    cfg, wrapper = load_model(args.config, args.checkpoint)
    metadata = make_metadata(cfg)
    export(wrapper, args.output, metadata['input_size'], metadata)
    report = {
        'onnx_path': str(args.output),
        'checkpoint_path': str(args.checkpoint),
        'config_path': str(args.config),
        'metadata': metadata,
    }
    if args.verify_images:
        report['parity'] = verify(wrapper, args.output, image_paths, args.verify_device,
                                  args.verify_threshold)
    report_path = args.output.with_suffix('.verification.json')
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
