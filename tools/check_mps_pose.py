"""Run a four-point GTR train step and COCO evaluation on Apple MPS."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torchvision.transforms.functional import to_tensor

from engine.core import YAMLConfig
from engine.data.dataset.coco_eval import CocoEvaluator
from engine.data.dataset.coco_pose_dataset import CocoPoseDetection
from engine.data.transforms.pose_transforms import PoseNormalize


def load_sample(dataset, index, normalizer, device):
    image, target = dataset.load_item(index)
    image, normalized = normalizer(to_tensor(image), target)
    normalized = {key: value.to(device) for key, value in normalized.items()}
    return image.unsqueeze(0).to(device), target, normalized


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/pose/synthetic_quad.yml')
    parser.add_argument('--data', type=Path, default=Path('dataset/synthetic_quad_pose'))
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        parser.error('Apple MPS is unavailable on this machine')

    device = torch.device('mps')
    cfg = YAMLConfig(args.config)
    # The smoke check intentionally starts from random weights.
    cfg.yaml_cfg['ViTAdapterSpatialSwiGLU']['skip_weights_warning'] = True
    model = cfg.model.to(device)
    criterion = cfg.criterion.to(device)
    postprocessor = cfg.postprocessor.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    normalizer = PoseNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_data = CocoPoseDetection(args.data / 'train', str(args.data / 'train.json'), None)
    val_data = CocoPoseDetection(args.data / 'val', str(args.data / 'val.json'), None)

    model.train()
    criterion.train()
    image, _, target = load_sample(train_data, 0, normalizer, device)
    output = model(image, targets=[target])
    assert output['pred_keypoints'].shape[-1] == 8
    losses = criterion(output, [target])
    loss = sum(losses.values())
    assert bool(torch.isfinite(loss).item())
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.mps.synchronize()
    optimizer.step()
    torch.mps.synchronize()

    model.eval()
    predictions = {}
    with torch.inference_mode():
        for index in range(len(val_data)):
            image, raw_target, _ = load_sample(val_data, index, normalizer, device)
            output = model(image)
            result = postprocessor(output, raw_target['orig_size'].reshape(1, 2).to(device))[0]
            predictions[int(raw_target['image_id'].item())] = {
                key: value.cpu() for key, value in result.items()
            }
    evaluator = CocoEvaluator(val_data.coco, ['keypoints'], verbose=False)
    evaluator.update(predictions)
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()
    stats = evaluator.coco_eval['keypoints'].stats
    report = {
        'device': str(device), 'train_loss': float(loss.detach().cpu()),
        'prediction_shape': list(output['pred_keypoints'].shape),
        'val_images': len(val_data), 'AP': float(stats[0]), 'AP50': float(stats[1]),
        'weights': 'random, one optimizer step',
    }
    (args.data / 'mps_evaluation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
