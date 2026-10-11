"""Random rectangular cutout augmentation, adapted from ecpose."""

import math
import numbers
import random

import PIL
import torch
from PIL import ImageDraw

from ...core import register


@register()
class RandomCutOut:
    """Draw ``n`` randomly sized black rectangles on an image.

    Each rectangle's width and height are sampled independently between the
    minimum and maximum ratios in ``scale``. Target annotations are returned
    unchanged.

    Args:
        p: Probability of applying cutout, in the range [0, 1].
        scale: ``(minimum, maximum)`` width and height ratios. Both values
            must be in [0, 0.5]. A list is also accepted for YAML configs.
        n: Number of rectangles to draw. Must be non-negative.
    """

    def __init__(self, p=0.5, scale=(0.0, 0.5), n=1):
        self.p = self._validate_float(p, "p", 0.0, 1.0)
        self.scale = self._validate_scale(scale)
        if not isinstance(n, numbers.Integral) or isinstance(n, bool):
            raise TypeError("n must be an integer")
        if n < 0:
            raise ValueError("n must be non-negative")
        self.n = int(n)

    @staticmethod
    def _validate_float(value, name, minimum, maximum):
        if not isinstance(value, numbers.Real) or isinstance(value, bool):
            raise TypeError(f"{name} must be a number")
        value = float(value)
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(f"{name} must be between {minimum} and {maximum}")
        return value

    @classmethod
    def _validate_scale(cls, scale):
        if not isinstance(scale, (tuple, list)) or len(scale) != 2:
            raise TypeError("scale must be a (minimum, maximum) pair")
        minimum = cls._validate_float(scale[0], "scale minimum", 0.0, 0.5)
        maximum = cls._validate_float(scale[1], "scale maximum", 0.0, 0.5)
        if minimum > maximum:
            raise ValueError("scale minimum must not exceed scale maximum")
        return minimum, maximum

    @staticmethod
    def _size(image):
        if isinstance(image, PIL.Image.Image):
            return image.width, image.height
        if isinstance(image, torch.Tensor):
            if image.ndim < 2:
                raise ValueError(
                    "RandomCutOut expects a tensor with at least 2 dimensions"
                )
            return image.shape[-1], image.shape[-2]
        raise TypeError(
            f"RandomCutOut expects a PIL image or tensor, got {type(image).__name__}"
        )

    @staticmethod
    def _draw(image, rectangles):
        if isinstance(image, PIL.Image.Image):
            output = image.copy()
            drawing = ImageDraw.Draw(output)
            for x, y, width, height in rectangles:
                drawing.rectangle(
                    (x, y, x + width - 1, y + height - 1), fill=0
                )
            return output

        output = image.clone()
        for x, y, width, height in rectangles:
            output[..., y:y + height, x:x + width] = 0
        return output

    def __call__(self, image, target=None):
        width, height = self._size(image)
        minimum, maximum = self.scale
        min_width = max(1, math.ceil(width * minimum))
        min_height = max(1, math.ceil(height * minimum))
        max_width = math.floor(width * maximum)
        max_height = math.floor(height * maximum)
        if (
            self.n == 0
            or min_width > max_width
            or min_height > max_height
            or random.random() >= self.p
        ):
            return image, target

        rectangles = []
        for _ in range(self.n):
            rectangle_width = random.randint(min_width, max_width)
            rectangle_height = random.randint(min_height, max_height)
            x = random.randint(0, width - rectangle_width)
            y = random.randint(0, height - rectangle_height)
            rectangles.append((x, y, rectangle_width, rectangle_height))

        return self._draw(image, rectangles), target

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(p={self.p}, scale={self.scale}, "
            f"n={self.n})"
        )
