"""Geometry and annotation checks for the ecpose pose augmentations."""

import random
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image, ImageDraw

from engine.data.transforms import (
    CutPaste, GaussianBlur, GaussianNoise, JPEGDeter, MotionBlur,
    RandomCutOut, RandomRotation,
)


def square_target(left, top, right, bottom):
    return {
        'keypoints': torch.tensor([[[left, top, 2], [right, top, 2],
                                    [right, bottom, 2], [left, bottom, 2]]],
                                  dtype=torch.float32),
        'boxes': torch.tensor([[left, top, right, bottom]], dtype=torch.float32),
        'area': torch.tensor([(right - left) * (bottom - top)], dtype=torch.float32),
        'labels': torch.tensor([0]),
        'iscrowd': torch.tensor([0]),
    }


class PoseAugmentationTest(unittest.TestCase):
    def test_rotation_keeps_pixels_and_annotations_aligned(self):
        image = Image.new('RGB', (80, 60))
        image.putpixel((60, 30), (255, 255, 255))
        target = square_target(50, 20, 60, 30)
        original_keypoints = target['keypoints'].clone()
        with patch('engine.data.transforms.pose_augmentations.random.uniform', return_value=90):
            rotated, result = RandomRotation(degrees=90, p=1)(image, target)

        self.assertEqual(rotated.size, image.size)
        self.assertEqual(result['keypoints'][0, 2, 2].item(), 2)
        torch.testing.assert_close(result['keypoints'][0, 2, :2], torch.tensor([40., 15.]))
        torch.testing.assert_close(target['keypoints'], original_keypoints)
        point = result['keypoints'][0, 2, :2].round().int().tolist()
        pixels = np.asarray(rotated)
        self.assertGreater(int(pixels[point[1]-2:point[1]+3,
                                      point[0]-2:point[0]+3].max()), 0)
        bounds = result['boxes'][0]
        for keypoint in result['keypoints'][0, :, :2]:
            self.assertTrue(bool((keypoint >= bounds[:2] - 1e-4).all()))
            self.assertTrue(bool((keypoint <= bounds[2:] + 1e-4).all()))

    def test_cutpaste_adds_aligned_object_without_mutating_source(self):
        source = Image.new('RGB', (64, 64))
        ImageDraw.Draw(source).rectangle((8, 8, 20, 20), fill=(255, 0, 0))
        source_target = square_target(8, 8, 20, 20)
        destination = Image.new('RGB', (64, 64))
        destination_target = square_target(40, 40, 52, 52)
        operation = CutPaste(p=1, max_pastes=1, max_trials=100)
        operation(source, source_target)
        random.seed(3)
        result_image, result_target = operation(destination, destination_target)

        self.assertEqual(result_target['keypoints'].shape, (2, 4, 3))
        self.assertEqual(result_target['boxes'].shape, (2, 4))
        self.assertEqual(result_target['labels'].shape, (2,))
        self.assertTrue(bool((result_target['keypoints'][1, :, 2] == 2).all()))
        self.assertGreater(int(np.asarray(result_image)[..., 0].max()), 0)
        self.assertEqual(destination_target['keypoints'].shape[0], 1)
        self.assertEqual(source_target['keypoints'].shape[0], 1)

    def test_image_only_effects_preserve_annotations(self):
        pattern = ((np.indices((64, 64)).sum(axis=0) % 2) * 255).astype(np.uint8)
        image = Image.fromarray(np.repeat(pattern[..., None], 3, axis=-1))
        target = square_target(8, 8, 20, 20)
        operations = [
            GaussianNoise(std=0.1, p=1), GaussianBlur(kernel_size=5, p=1),
            MotionBlur(kernel_size=5, angle=0, p=1),
            JPEGDeter(quality=30, p=1),
            RandomCutOut(p=1, scale=(0.25, 0.25), n=1),
        ]
        for operation in operations:
            with self.subTest(operation=type(operation).__name__):
                changed, result_target = operation(image, target)
                self.assertIs(result_target, target)
                self.assertEqual(changed.size, image.size)
                self.assertTrue(np.any(np.asarray(changed) != np.asarray(image)))


if __name__ == '__main__':
    unittest.main()
