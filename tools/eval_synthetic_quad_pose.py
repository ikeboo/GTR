"""Check four-point data, matching, loss, postprocessing, and COCO OKS AP.

Uses annotation coordinates as oracle predictions, so AP here validates the
pipeline rather than measuring a trained GTR checkpoint.
"""

import argparse
import json
import sys
import types
from pathlib import Path

import torch
from torchvision.transforms.functional import to_tensor

# This check exercises the pose data/loss/evaluation path without loading the
# backbone's optional CUDA/fla operators. The real training entry point imports
# and registers the complete engine as usual.
ROOT = Path(__file__).resolve().parents[1]
for name, directory in [
    ('engine', 'engine'),
    ('engine.data', 'engine/data'),
    ('engine.data.dataset', 'engine/data/dataset'),
    ('engine.data.transforms', 'engine/data/transforms'),
    ('engine.gtr', 'engine/gtr'),
    ('engine.gtr.pose', 'engine/gtr/pose'),
]:
    package = types.ModuleType(name)
    package.__path__ = [str(ROOT / directory)]
    sys.modules[name] = package
sys.modules['engine.data'].DataLoader = torch.utils.data.DataLoader

from engine.data.dataset.coco_eval import CocoEvaluator
from engine.data.dataset.coco_pose_dataset import CocoPoseDetection
from engine.data.transforms.pose_transforms import PoseNormalize
from engine.gtr.pose.keypoint_loss import OKSLoss
from engine.gtr.pose.criterion import GTRPoseCriterion
from engine.gtr.pose.matcher import GTRPoseHungarianMatcher
from engine.gtr.pose.postprocessor import GTRPosePostProcessor


def evaluate(coco, predictions):
    evaluator = CocoEvaluator(coco, ['keypoints'], verbose=False)
    evaluator.update(predictions)
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()
    return list(evaluator.coco_eval['keypoints'].stats[:2])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('dataset/synthetic_quad_pose'))
    args = parser.parse_args()

    dataset = CocoPoseDetection(args.data / 'val', str(args.data / 'val.json'), None)
    normalizer = PoseNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    postprocessor = GTRPosePostProcessor(num_select=1, num_body_points=4)
    matcher = GTRPoseHungarianMatcher(num_body_points=4)
    oks_loss = OKSLoss(num_keypoints=4, linear=True)
    criterion = GTRPoseCriterion(num_classes=2, matcher=matcher,
                                 weight_dict={'loss_vfl': 2.0, 'loss_keypoints': 10.0,
                                              'loss_oks': 4.0},
                                 losses=['vfl', 'keypoints'], num_body_points=4)
    exact, shifted = {}, {}
    losses = []
    criterion_loss = None
    criterion_grad_norm = None

    for index in range(len(dataset)):
        image, target = dataset.load_item(index)
        assert target['keypoints'].shape == (1, 4, 3)
        _, normalized = normalizer(to_tensor(image), target)
        coordinates = normalized['keypoints'][0, :8]
        logits = torch.tensor([[[-8.0, 8.0]]])
        outputs = {'pred_logits': logits, 'pred_keypoints': coordinates.reshape(1, 1, 8)}
        matched = matcher(outputs, [normalized])
        assert matched[0][0].tolist() == [0] and matched[0][1].tolist() == [0]
        if criterion_loss is None:
            training_logits = logits.clone().requires_grad_()
            training_points = (outputs['pred_keypoints'].clone() + 1 / 128).requires_grad_()
            training_prediction = {'pred_logits': training_logits,
                                   'pred_keypoints': training_points}
            training_outputs = {**training_prediction, 'aux_outputs': [],
                                'aux_pre_outputs': training_prediction,
                                'aux_interm_outputs': []}
            criterion_losses = criterion(training_outputs, [normalized])
            assert all(torch.isfinite(value).all() for value in criterion_losses.values())
            total_loss = sum(criterion_losses.values())
            total_loss.backward()
            assert torch.isfinite(training_points.grad).all()
            assert torch.isfinite(training_logits.grad).all()
            criterion_loss = float(total_loss.detach())
            criterion_grad_norm = float(training_points.grad.norm())
        losses.append(float(oks_loss(coordinates[None], coordinates[None],
                                     normalized['keypoints'][:, 8:], normalized['area'])))
        result = postprocessor(outputs, target['orig_size'].reshape(1, 2))[0]
        image_id = int(target['image_id'].item())
        exact[image_id] = result
        shifted[image_id] = {key: value.clone() for key, value in result.items()}
        shifted[image_id]['keypoints'][:, 0::3] += 5
        shifted[image_id]['keypoints'][:, 1::3] += 5

    exact_ap, exact_ap50 = evaluate(dataset.coco, exact)
    shifted_ap, shifted_ap50 = evaluate(dataset.coco, shifted)
    report = {
        'images': len(dataset), 'keypoints': dataset.num_keypoints,
        'oracle_oks_mean': sum(losses) / len(losses),
        'criterion_loss_1px': criterion_loss,
        'criterion_grad_norm_1px': criterion_grad_norm,
        'oracle_AP': exact_ap, 'oracle_AP50': exact_ap50,
        'shifted_5px_AP': shifted_ap, 'shifted_5px_AP50': shifted_ap50,
    }
    print(json.dumps(report, indent=2))
    assert exact_ap > 0.99 and shifted_ap < exact_ap
    (args.data / 'evaluation.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
