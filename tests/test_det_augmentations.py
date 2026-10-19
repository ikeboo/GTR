"""Image/box alignment and custom detection configuration regression checks."""

import copy
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

from engine.core import YAMLConfig
from engine.core.yaml_utils import load_config
from engine.data._misc import BoundingBoxes, convert_to_tv_tensor
from engine.data.dataloader import BatchImageCollateFunction
from engine.data.transforms import (
    Compose, GaussianBlur, GaussianNoise, MotionBlur,
    RandomHSV, RandomPerspective, RandomVerticalFlip,
)


def target_for(boxes):
    boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    count = len(boxes)
    return {
        'boxes': convert_to_tv_tensor(boxes, key='boxes', spatial_size=(80, 100)),
        'labels': torch.arange(count),
        'area': (boxes[:, 2:] - boxes[:, :2]).prod(dim=1),
        'iscrowd': torch.arange(count),
        'image_id': torch.tensor([123]),
        'orig_size': torch.tensor([80, 100]),
    }


class DetectionAugmentationTest(unittest.TestCase):
    def test_saturation_gain_preserves_white_and_annotations(self):
        pixels = np.array([[(100, 150, 200), (255, 255, 255)]], dtype=np.uint8)
        image = Image.fromarray(pixels)
        target = target_for([[10, 10, 30, 30]])
        with patch('engine.data.transforms.det_augmentations.random.uniform',
                   side_effect=[0, 0.5, 1]):
            changed, result = RandomHSV(sgain=0.5)(image, target)
        hsv_before = cv2.cvtColor(pixels, cv2.COLOR_RGB2HSV)
        hsv_after = cv2.cvtColor(np.asarray(changed), cv2.COLOR_RGB2HSV)
        self.assertLessEqual(abs(int(hsv_after[0, 0, 1]) - int(hsv_before[0, 0, 1]) * 0.5), 2)
        np.testing.assert_allclose(hsv_after[..., (0, 2)], hsv_before[..., (0, 2)], atol=1)
        np.testing.assert_array_equal(np.asarray(changed)[0, 1], (255, 255, 255))
        self.assertIs(result, target)

    def test_disabled_hsv_is_identity(self):
        image = Image.new('RGB', (100, 80))
        target = target_for([])
        changed, result = RandomHSV()(image, target)
        self.assertIs(changed, image)
        self.assertIs(result, target)

    def test_rotation_aligns_pixels_and_boxes_without_fit_scaling(self):
        image = Image.new('RGB', (100, 80))
        ImageDraw.Draw(image).rectangle((60, 20, 79, 39), fill='white')
        target = target_for([[60, 20, 80, 40]])
        with patch('engine.data.transforms.det_augmentations.random.uniform',
                   side_effect=[0, 0, 90, 1, 0, 0]):
            changed, result = RandomPerspective(degrees=90)(image, target)
        torch.testing.assert_close(result['boxes'].as_subclass(torch.Tensor),
                                   torch.tensor([[30., 10., 50., 30.]]))
        self.assertGreater(int(np.asarray(changed)[20, 40].min()), 250)
        self.assertIsInstance(result['boxes'], BoundingBoxes)
        self.assertEqual(changed.size, image.size)
        torch.testing.assert_close(target['boxes'].as_subclass(torch.Tensor),
                                   torch.tensor([[60., 20., 80., 40.]]))

    def test_scale_extremes_keep_center_with_no_translation(self):
        image = Image.new('RGB', (100, 80))
        for scale in (0.7, 1.3):
            with self.subTest(scale=scale), patch(
                'engine.data.transforms.det_augmentations.random.uniform',
                side_effect=[0, 0, 0, scale, 0, 0],
            ):
                _, result = RandomPerspective(scale=0.3)(image, target_for([[40, 30, 60, 50]]))
                torch.testing.assert_close(result['boxes'].as_subclass(torch.Tensor),
                                           torch.tensor([[50 - 10 * scale, 40 - 10 * scale,
                                                          50 + 10 * scale, 40 + 10 * scale]]))
                torch.testing.assert_close(result['area'], torch.tensor([400 * scale ** 2]))

    def test_perspective_projects_corners_and_image_with_same_matrix(self):
        image = Image.new('RGB', (100, 80))
        ImageDraw.Draw(image).rectangle((60, 20, 79, 39), fill='white')
        coefficient = 0.001
        with patch('engine.data.transforms.det_augmentations.random.uniform',
                   side_effect=[coefficient, 0, 0, 1, 0, 0]):
            changed, result = RandomPerspective(perspective=coefficient)(
                image, target_for([[60, 20, 80, 40]]))
        expected = torch.tensor([[50 + 10 / 1.01, 40 - 20 / 1.01,
                                  50 + 30 / 1.03, 40.]])
        torch.testing.assert_close(result['boxes'].as_subclass(torch.Tensor), expected)
        self.assertGreater(int(np.asarray(changed)[30, 70].min()), 250)

    def test_clipping_removes_instances_and_keeps_fields_aligned(self):
        image = Image.new('RGB', (100, 80))
        target = target_for([[10, 10, 30, 30], [80, 10, 95, 30], [39, 50, 41, 70]])
        operation = RandomPerspective()
        matrix = np.eye(3)
        matrix[0, 2] = 60
        with patch.object(operation, '_get_matrix', return_value=(matrix, 1)):
            _, result = operation(image, target)
        torch.testing.assert_close(result['boxes'].as_subclass(torch.Tensor),
                                   torch.tensor([[70., 10., 90., 30.]]))
        for field in ('labels', 'iscrowd'):
            torch.testing.assert_close(result[field], torch.tensor([0]))
        torch.testing.assert_close(result['area'], torch.tensor([400.]))
        self.assertEqual(result['image_id'].item(), 123)
        self.assertEqual(len(target['boxes']), 3)

    def test_empty_annotations_and_dataset_passthrough(self):
        image = Image.new('RGB', (100, 80))
        dataset = SimpleNamespace(epoch=0)
        for operation in (RandomHSV(sgain=0.5), RandomPerspective(perspective=0.00005)):
            with self.subTest(operation=type(operation).__name__):
                changed, result, same_dataset = operation((image, target_for([]), dataset))
                self.assertEqual(result['boxes'].shape, (0, 4))
                self.assertEqual(result['area'].shape, (0,))
                self.assertEqual(changed.size, image.size)
                self.assertIs(same_dataset, dataset)

    def test_vertical_flip_updates_boxes(self):
        image = Image.new('RGB', (100, 80))
        _, result = RandomVerticalFlip(p=1)(image, target_for([[10, 10, 30, 30]]))
        torch.testing.assert_close(result['boxes'].as_subclass(torch.Tensor),
                                   torch.tensor([[10., 50., 30., 70.]]))

    def test_invalid_parameters_and_unsupported_annotations(self):
        for kwargs in ({'scale': 1}, {'perspective': float('nan')}, {'translate': -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RandomPerspective(**kwargs)
        with self.assertRaises(ValueError):
            RandomHSV(sgain=1.5)
        target = target_for([])
        target['masks'] = torch.empty(0, 80, 100)
        with self.assertRaises(ValueError):
            RandomPerspective()(Image.new('RGB', (100, 80)), target)

    def test_blur_and_noise_preserve_detection_target_and_dataset(self):
        pattern = ((np.indices((80, 100)).sum(axis=0) % 2) * 255).astype(np.uint8)
        image = Image.fromarray(np.repeat(pattern[..., None], 3, axis=-1))
        target = target_for([[10, 10, 30, 30]])
        dataset = SimpleNamespace(epoch=0)
        for kind, kwargs in ((MotionBlur, {'kernel_size': 5, 'angle': 0}),
                             (GaussianBlur, {'kernel_size': 5, 'sigma': 1.0}),
                             (GaussianNoise, {'std': 0.1})):
            for probability in (0, 1):
                with self.subTest(kind=kind.__name__, p=probability):
                    operation = kind(p=probability, **kwargs)
                    changed, same_target, same_dataset = operation((image, target, dataset))
                    self.assertIs(same_target, target)
                    self.assertIs(same_dataset, dataset)
                    self.assertEqual(changed.size, image.size)
                    if probability:
                        self.assertTrue(np.any(np.asarray(changed) != np.asarray(image)))
                    else:
                        self.assertIs(changed, image)
                    _, same_target, same_dataset = operation(image, target, dataset)
                    self.assertIs(same_target, target)
                    self.assertIs(same_dataset, dataset)

    def test_custom_config_pipeline_and_no_aug_schedule(self):
        root = Path(__file__).resolve().parents[1]
        cfg = load_config(str(root / 'configs/det/gtr_s_custom.yml'))
        transforms = cfg['train_dataloader']['dataset']['transforms']
        pipeline = Compose(**{key: value for key, value in transforms.items() if key != 'type'})
        names = [type(op).__name__ for op in pipeline.transforms]
        self.assertEqual(names, ['Resize', 'RandomPerspective', 'RandomHSV', 'ColorJitter',
                                 'MotionBlur', 'GaussianBlur', 'GaussianNoise',
                                 'RandomVerticalFlip', 'RandomHorizontalFlip',
                                 'ConvertPILImage', 'Normalize', 'ConvertBoxes'])
        operations = dict(zip(names, pipeline.transforms))
        geometry, hsv, contrast, vertical, horizontal = (
            operations[name] for name in ('RandomPerspective', 'RandomHSV', 'ColorJitter',
                                         'RandomVerticalFlip', 'RandomHorizontalFlip'))
        self.assertEqual((geometry.degrees, geometry.scale, geometry.perspective, geometry.translate),
                         (15, 0.3, 0.00005, 0))
        hsv_config = next(op for op in transforms['ops'] if op['type'] == 'RandomHSV')
        self.assertEqual((hsv.hgain, hsv.sgain, hsv.vgain),
                         (hsv_config['hgain'], hsv_config['sgain'], hsv_config['vgain']))
        self.assertEqual(contrast.contrast, (0.5, 1.5))
        motion, blur, noise = (operations[name] for name in ('MotionBlur', 'GaussianBlur', 'GaussianNoise'))
        self.assertEqual((motion.p, blur.p, noise.p), (0.2, 0.3, 0.5))
        self.assertEqual(noise.std, 0.05)
        self.assertEqual((vertical.p, horizontal.p, pipeline.mosaic_prob), (0.5, 0.5, 0))
        collate_cfg = cfg['train_dataloader']['collate_fn']
        collate = BatchImageCollateFunction(**{k: v for k, v in collate_cfg.items() if k != 'type'})
        self.assertIsNone(collate.scales)
        self.assertEqual((collate.mixup_prob, collate.copyblend_prob), (0, 0))
        image = Image.new('RGB', (100, 80), (100, 150, 200))
        for epoch, expected_calls in ((0, 1), (57, 1), (58, 0), (59, 0)):
            with (self.subTest(epoch=epoch),
                  patch.object(geometry, 'forward', wraps=geometry.forward) as geo,
                  patch.object(hsv, 'forward', wraps=hsv.forward) as color,
                  patch.object(contrast, 'forward', wraps=contrast.forward) as jitter,
                  patch.object(motion, '_apply_image', wraps=motion._apply_image) as motion_call,
                  patch.object(blur, '_apply_image', wraps=blur._apply_image) as blur_call,
                  patch.object(noise, '_apply_image', wraps=noise._apply_image) as noise_call):
                random.seed(42)
                torch.manual_seed(42)
                tensor, target, dataset = pipeline(image, target_for([[20, 20, 80, 60]]),
                                                    SimpleNamespace(epoch=epoch))
                self.assertEqual(tensor.shape, (3, 640, 640))
                self.assertTrue(torch.isfinite(tensor).all())
                self.assertTrue(((target['boxes'] >= 0) & (target['boxes'] <= 1)).all())
                self.assertEqual(geo.call_count, expected_calls)
                self.assertEqual(color.call_count, expected_calls)
                self.assertEqual(jitter.call_count, expected_calls)
                for effect in (motion_call, blur_call, noise_call):
                    self.assertEqual(effect.call_count, expected_calls)
                self.assertEqual(dataset.epoch, epoch)

    def test_dataset_setting_edits_reach_standard_and_mps_transforms(self):
        root = Path(__file__).resolve().parents[1]
        dataset_path = root / 'configs/dataset/custom_detection.yml'
        model_path = root / 'configs/det/gtr_s_custom.yml'
        dataset_cfg = yaml.safe_load(dataset_path.read_text())
        changes = {
            'RandomPerspective': {'degrees': 9, 'translate': 0.1, 'scale': 0.2, 'perspective': 0.00002},
            'RandomHSV': {'sgain': 0.2},
            'ColorJitter': {'contrast': [0.8, 1.2]},
            'MotionBlur': {'p': 0.1},
            'GaussianBlur': {'p': 0.8},
            'GaussianNoise': {'std': 0.1, 'p': 0.7},
            'RandomVerticalFlip': {'p': 0.25},
            'RandomHorizontalFlip': {'p': 0.75},
        }
        dataset_transforms = dataset_cfg['train_dataloader']['dataset']['transforms']
        for op in dataset_transforms['ops']:
            op.update(changes.get(op['type'], {}))
        dataset_transforms['stop_epoch'] = 3
        dataset_cfg['train_dataloader']['collate_fn']['stop_epoch'] = 3
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            edited_dataset = directory / 'dataset.yml'
            edited_dataset.write_text(yaml.safe_dump(dataset_cfg))
            model_cfg = yaml.safe_load(model_path.read_text())
            model_cfg['__include__'] = [
                str(edited_dataset) if Path(path).name == dataset_path.name
                else str((model_path.parent / path).resolve())
                for path in model_cfg['__include__']
            ]
            edited_model = directory / 'model.yml'
            edited_model.write_text(yaml.safe_dump(model_cfg))
            mps_cfg = yaml.safe_load((root / 'configs/det/gtr_s_custom_mps.yml').read_text())
            mps_cfg['__include__'] = [str(edited_model)]
            edited_mps = directory / 'mps.yml'
            edited_mps.write_text(yaml.safe_dump(mps_cfg))
            for path in (edited_model, edited_mps):
                with self.subTest(config=path.name):
                    cfg = YAMLConfig(str(path))
                    transforms = cfg.yaml_cfg['train_dataloader']['dataset']['transforms']
                    pipeline = Compose(**{k: copy.deepcopy(v) for k, v in transforms.items() if k != 'type'})
                    self.assertEqual(pipeline.stop_epoch, 3)
                    for operation in pipeline.transforms:
                        for name, expected in changes.get(type(operation).__name__, {}).items():
                            if name == 'contrast':
                                expected = tuple(expected)
                            self.assertEqual(getattr(operation, name), expected)
                    collate_cfg = cfg.yaml_cfg['train_dataloader']['collate_fn']
                    collate = BatchImageCollateFunction(**{k: v for k, v in collate_cfg.items() if k != 'type'})
                    self.assertEqual(collate.stop_epoch, 3)
                    self.assertIsNone(collate.scales)


if __name__ == '__main__':
    unittest.main()
