"""YOLO-style augmentations for PIL RGB images and pixel-space detection boxes.

Place these before ConvertPILImage / ConvertBoxes. Both accept Compose's
(image, target, dataset) tuple as well as an (image, target) pair.
"""

import math
import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn

from ...core import register
from .._misc import BoundingBoxes, convert_to_tv_tensor


def _sample(inputs):
    sample = inputs if len(inputs) > 1 else inputs[0]
    if len(sample) not in (2, 3):
        raise ValueError('Expected (image, target) or (image, target, dataset).')
    if not isinstance(sample[0], Image.Image) or sample[0].mode != 'RGB':
        raise TypeError('Detection augmentations require a PIL RGB image before ConvertPILImage.')
    return sample


@register()
class RandomHSV(nn.Module):
    """Uniform HSV gains; sgain=0.5 multiplies saturation by 0.5 to 1.5.

    Hue uses an additive offset in OpenCV's 0..179 range, while saturation
    and value use multiplicative gains. Zero gains disable their channels.
    """

    def __init__(self, hgain=0.0, sgain=0.0, vgain=0.0):
        super().__init__()
        for name, gain in (('hgain', hgain), ('sgain', sgain), ('vgain', vgain)):
            if not math.isfinite(gain) or not 0 <= gain <= 1:
                raise ValueError(f'{name} must be finite and between 0 and 1.')
        self.hgain, self.sgain, self.vgain = hgain, sgain, vgain

    def forward(self, *inputs):
        image, target, *rest = _sample(inputs)
        if not (self.hgain or self.sgain or self.vgain):
            return (image, target, *rest)

        hue, saturation, value = cv2.split(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2HSV))
        hue_offset = random.uniform(-self.hgain, self.hgain) * 180
        saturation_gain = random.uniform(1 - self.sgain, 1 + self.sgain)
        value_gain = random.uniform(1 - self.vgain, 1 + self.vgain)
        levels = np.arange(256, dtype=np.float64)
        hue_lut = ((levels + hue_offset) % 180).astype(np.uint8)
        saturation_lut = np.clip(levels * saturation_gain, 0, 255).astype(np.uint8)
        value_lut = np.clip(levels * value_gain, 0, 255).astype(np.uint8)
        hsv = cv2.merge((cv2.LUT(hue, hue_lut), cv2.LUT(saturation, saturation_lut),
                         cv2.LUT(value, value_lut)))
        image = Image.fromarray(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
        return (image, target, *rest)


@register()
class RandomPerspective(nn.Module):
    """Joint rotation, scale, perspective and translation on a fixed canvas.

    Rotation is uniform in [-degrees, degrees], scale in [1-scale, 1+scale],
    and the two projective coefficients in [-perspective, perspective].
    Translation is a fraction of each image dimension; zero keeps the center
    fixed. Boxes are projected with the same homography as the image, clipped,
    and filtered using YOLO's detection criteria (>2 px sides, >10% retained
    scaled area, aspect ratio <100). Labels, area and iscrowd stay aligned.
    This transform is for axis-aligned detection, without masks or keypoints.
    """

    def __init__(self, degrees=0.0, translate=0.0, scale=0.0, perspective=0.0, fill=114):
        super().__init__()
        for name, value in (('degrees', degrees), ('translate', translate),
                            ('scale', scale), ('perspective', perspective)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and non-negative.')
        if scale >= 1 or translate > 1:
            raise ValueError('scale must be <1 and translate must be <=1.')
        self.degrees, self.translate = degrees, translate
        self.scale, self.perspective = scale, perspective
        self.fill = (fill, fill, fill) if isinstance(fill, (int, float)) else tuple(fill)

    def _get_matrix(self, width, height):
        center = np.eye(3, dtype=np.float64)
        center[:2, 2] = (-width / 2, -height / 2)
        projective = np.eye(3, dtype=np.float64)
        projective[2, :2] = [random.uniform(-self.perspective, self.perspective),
                              random.uniform(-self.perspective, self.perspective)]
        angle = random.uniform(-self.degrees, self.degrees)
        scale = random.uniform(1 - self.scale, 1 + self.scale)
        rotation = np.eye(3, dtype=np.float64)
        rotation[:2] = cv2.getRotationMatrix2D((0, 0), angle, scale)
        translation = np.eye(3, dtype=np.float64)
        translation[:2, 2] = (
            (0.5 + random.uniform(-self.translate, self.translate)) * width,
            (0.5 + random.uniform(-self.translate, self.translate)) * height,
        )
        return translation @ rotation @ projective @ center, scale

    @staticmethod
    def _transform_target(target, matrix, scale, width, height):
        boxes = target['boxes']
        if isinstance(boxes, BoundingBoxes) and boxes.format.value != 'XYXY':
            raise ValueError('RandomPerspective requires pixel-space XYXY boxes.')
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError('RandomPerspective requires boxes with shape (N, 4).')
        # Strip TV metadata during arithmetic, then restore it for later flips.
        original = boxes.as_subclass(torch.Tensor).to(dtype=torch.float32)
        corners = original[:, [0, 1, 2, 1, 2, 3, 0, 3]].reshape(-1, 4, 2)
        homogeneous = torch.cat((corners, torch.ones_like(corners[..., :1])), dim=-1)
        projected = homogeneous @ torch.as_tensor(matrix, dtype=original.dtype,
                                                   device=original.device).T
        xy = projected[..., :2] / projected[..., 2:]
        warped = torch.cat((xy.amin(dim=1), xy.amax(dim=1)), dim=1)
        finite = torch.isfinite(warped).all(dim=1)
        warped[:, 0::2].clamp_(0, width)
        warped[:, 1::2].clamp_(0, height)
        wh = warped[:, 2:] - warped[:, :2]
        original_wh = (original[:, 2:] - original[:, :2]) * scale
        area = wh.prod(dim=1)
        aspect = torch.maximum(wh[:, 0] / (wh[:, 1] + 1e-16),
                               wh[:, 1] / (wh[:, 0] + 1e-16))
        keep = (finite & (wh > 2).all(dim=1) & (aspect < 100)
                & (area / (original_wh.prod(dim=1) + 1e-16) > 0.1))
        result = target.copy()
        for key in ('labels', 'area', 'iscrowd'):
            if key in result:
                result[key] = result[key][keep]
        if 'area' in result:
            result['area'] = area[keep].to(dtype=target['area'].dtype)
        result['boxes'] = convert_to_tv_tensor(warped[keep], key='boxes',
                                               box_format='XYXY', spatial_size=(height, width))
        return result

    def forward(self, *inputs):
        image, target, *rest = _sample(inputs)
        if any(key in target for key in ('masks', 'keypoints', 'polygons')):
            raise ValueError('RandomPerspective supports detection boxes without masks/keypoints/polygons.')
        width, height = image.size
        matrix, scale = self._get_matrix(width, height)
        if self.perspective:
            warped = cv2.warpPerspective(np.asarray(image), matrix, (width, height),
                                         flags=cv2.INTER_LINEAR, borderValue=self.fill)
        else:
            warped = cv2.warpAffine(np.asarray(image), matrix[:2], (width, height),
                                    flags=cv2.INTER_LINEAR, borderValue=self.fill)
        target = self._transform_target(target, matrix, scale, width, height)
        return (Image.fromarray(warped), target, *rest)
