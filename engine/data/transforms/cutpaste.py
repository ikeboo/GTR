"""Cut-and-paste augmentation for four-corner objects, adapted from ecpose."""

import copy
import math
import numbers
import random

import cv2
import numpy as np
import PIL
import torch
from PIL import Image, ImageDraw

from ...core import register


@register()
class CutPaste:
    """Paste four-corner objects from previously seen images.

    The transform uses the four keypoints of an object as a polygon mask, so
    pixels outside the annotated license plate are not copied. Source objects
    are taken only from earlier images kept in a bounded cache. A proposed
    paste is rejected when its polygon IoU with any object already in the
    destination image is greater than ``iou_threshold``.

    This transform is intended to run while images are still PIL images (that
    is, before ``ToTensor``).

    Args:
        p: Probability of trying the augmentation. Samples are cached even
            when augmentation is skipped.
        iou_threshold: Maximum allowed polygon IoU. The default is 0.1.
        max_pastes: Maximum number of objects pasted into one image.
        max_trials: Number of random destinations tried for each object.
        max_cached_images: Maximum number of earlier source images retained.
    """

    def __init__(
        self,
        p=0.5,
        iou_threshold=0.1,
        max_pastes=1,
        max_trials=50,
        max_cached_images=50,
    ):
        self.p = self._probability(p, "p")
        self.iou_threshold = self._probability(
            iou_threshold, "iou_threshold"
        )
        self.max_pastes = self._positive_int(max_pastes, "max_pastes")
        self.max_trials = self._positive_int(max_trials, "max_trials")
        self.max_cached_images = self._positive_int(
            max_cached_images, "max_cached_images"
        )
        self._cache = []

    @staticmethod
    def _probability(value, name):
        if not isinstance(value, numbers.Real) or isinstance(value, bool):
            raise TypeError(f"{name} must be a number")
        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
        return value

    @staticmethod
    def _positive_int(value, name):
        if not isinstance(value, numbers.Integral) or isinstance(value, bool):
            raise TypeError(f"{name} must be an integer")
        if value < 1:
            raise ValueError(f"{name} must be at least 1")
        return int(value)

    @staticmethod
    def _ordered_polygon(points):
        """Return a convex quadrilateral suitable for masking and IoU."""
        points = np.ascontiguousarray(
            np.asarray(points, dtype=np.float32).reshape(-1, 2)
        )
        if points.shape != (4, 2) or not np.isfinite(points).all():
            return None

        hull = cv2.convexHull(points).reshape(-1, 2)
        if hull.shape != (4, 2) or cv2.contourArea(hull) <= 0:
            return None
        return hull.astype(np.float32, copy=False)

    @classmethod
    def _polygon_iou(cls, first, second):
        """Compute IoU between two convex four-point polygons."""
        first = cls._ordered_polygon(first)
        second = cls._ordered_polygon(second)
        if first is None or second is None:
            return 0.0

        first_area = float(cv2.contourArea(first))
        second_area = float(cv2.contourArea(second))
        intersection, _ = cv2.intersectConvexConvex(first, second)
        union = first_area + second_area - float(intersection)
        return float(intersection) / union if union > 0 else 0.0

    @classmethod
    def _polygons_from_target(cls, target):
        keypoints = target.get("keypoints") if target is not None else None
        if keypoints is None:
            return []
        if not isinstance(keypoints, torch.Tensor):
            keypoints = torch.as_tensor(keypoints)
        if keypoints.ndim != 3 or keypoints.shape[-2:] not in ((4, 2), (4, 3)):
            raise ValueError(
                "CutPaste expects keypoints with shape (objects, 4, 2 or 3)"
            )

        polygons = []
        for index, keypoint in enumerate(keypoints.detach().cpu()):
            if keypoint.shape[-1] == 3 and not torch.all(keypoint[:, 2] > 0):
                continue
            polygon = cls._ordered_polygon(keypoint[:, :2].numpy())
            if polygon is not None:
                polygons.append((index, polygon))
        return polygons

    def _add_to_cache(self, image, target):
        if self._polygons_from_target(target):
            self._cache.append(
                {"image": image.copy(), "target": copy.deepcopy(target)}
            )
            if len(self._cache) > self.max_cached_images:
                self._cache.pop(0)

    def _source_objects(self):
        objects = []
        for item in self._cache:
            for index, polygon in self._polygons_from_target(item["target"]):
                objects.append(
                    {
                        "image": item["image"],
                        "target": item["target"],
                        "index": index,
                        "polygon": polygon,
                    }
                )
        return objects

    @staticmethod
    def _patch_bounds(polygon, image):
        left = max(0, int(math.floor(float(polygon[:, 0].min()))))
        top = max(0, int(math.floor(float(polygon[:, 1].min()))))
        right = min(
            image.width, int(math.ceil(float(polygon[:, 0].max()))) + 1
        )
        bottom = min(
            image.height, int(math.ceil(float(polygon[:, 1].max()))) + 1
        )
        return left, top, right, bottom

    @staticmethod
    def _append_like(target, key, value):
        reference = target.get(key)
        if not isinstance(reference, torch.Tensor):
            return
        value = torch.as_tensor(
            value, dtype=reference.dtype, device=reference.device
        ).reshape((1,) + tuple(reference.shape[1:]))
        target[key] = torch.cat((reference, value), dim=0)

    @classmethod
    def _append_annotation(
        cls, target, source_target, source_index, keypoints, polygon, mask
    ):
        cls._append_like(target, "keypoints", keypoints)
        cls._append_like(
            target,
            "boxes",
            [
                polygon[:, 0].min(),
                polygon[:, 1].min(),
                polygon[:, 0].max(),
                polygon[:, 1].max(),
            ],
        )
        cls._append_like(target, "area", float(cv2.contourArea(polygon)))

        for key, default in (("labels", 0), ("iscrowd", 0)):
            source_value = source_target.get(key)
            if (
                isinstance(source_value, torch.Tensor)
                and source_index < len(source_value)
            ):
                value = source_value[source_index]
            else:
                value = default
            cls._append_like(target, key, value)

        if isinstance(target.get("masks"), torch.Tensor):
            cls._append_like(target, "masks", mask)
        if isinstance(target.get("occlusion_ratio"), torch.Tensor):
            reference = target["occlusion_ratio"]
            cls._append_like(
                target,
                "occlusion_ratio",
                torch.zeros(
                    reference.shape[1:],
                    dtype=reference.dtype,
                    device=reference.device,
                ),
            )
        if isinstance(target.get("vis_ratio"), torch.Tensor):
            reference = target["vis_ratio"]
            cls._append_like(
                target,
                "vis_ratio",
                torch.ones(
                    reference.shape[1:],
                    dtype=reference.dtype,
                    device=reference.device,
                ),
            )
        if isinstance(target.get("mixup"), torch.Tensor):
            cls._append_like(target, "mixup", 1.0)

    @staticmethod
    def _translated_keypoints(source_target, source_index, offset):
        keypoints = source_target["keypoints"][source_index].clone()
        visible = (
            keypoints[:, 2] > 0
            if keypoints.shape[-1] == 3
            else torch.ones(
                len(keypoints), dtype=torch.bool, device=keypoints.device
            )
        )
        delta = keypoints.new_tensor(offset)
        keypoints[visible, :2] += delta
        return keypoints

    def _paste(self, destination, target, source, occupied):
        polygon = source["polygon"]
        source_image = source["image"]
        if (
            np.any(polygon[:, 0] < 0)
            or np.any(polygon[:, 0] > source_image.width)
            or np.any(polygon[:, 1] < 0)
            or np.any(polygon[:, 1] > source_image.height)
        ):
            return False
        left, top, right, bottom = self._patch_bounds(polygon, source_image)
        patch_width, patch_height = right - left, bottom - top
        if (
            patch_width <= 0
            or patch_height <= 0
            or patch_width > destination.width
            or patch_height > destination.height
        ):
            return False

        placement = None
        for _ in range(self.max_trials):
            x = random.randint(0, destination.width - patch_width)
            y = random.randint(0, destination.height - patch_height)
            candidate = polygon + np.array([x - left, y - top], np.float32)
            if all(
                self._polygon_iou(candidate, other) <= self.iou_threshold
                for other in occupied
            ):
                placement = x, y, candidate
                break
        if placement is None:
            return False

        x, y, candidate = placement
        patch = source_image.crop((left, top, right, bottom))
        if patch.mode != destination.mode:
            patch = patch.convert(destination.mode)
        patch_polygon = polygon - np.array([left, top], np.float32)
        alpha = Image.new("L", (patch_width, patch_height), 0)
        ImageDraw.Draw(alpha).polygon(
            [tuple(point) for point in patch_polygon], fill=255
        )
        destination.paste(patch, (x, y), alpha)

        full_mask = np.zeros((destination.height, destination.width), dtype=np.uint8)
        mask_polygon = np.ascontiguousarray(
            np.rint(candidate).astype(np.int32)
        )
        cv2.fillConvexPoly(full_mask, mask_polygon, 1)
        source_target = source["target"]
        keypoints = self._translated_keypoints(
            source_target, source["index"], (x - left, y - top)
        )
        self._append_annotation(
            target,
            source_target,
            source["index"],
            keypoints,
            candidate,
            full_mask,
        )
        occupied.append(candidate)
        return True

    def __call__(self, image, target):
        if not isinstance(image, PIL.Image.Image):
            raise TypeError(
                f"CutPaste expects a PIL image, got {type(image).__name__}"
            )
        if target is None:
            return image, target

        # Build the source pool before caching this sample. This guarantees
        # that an object is always pasted from a different image.
        source_objects = self._source_objects()
        self._add_to_cache(image, target)

        if not source_objects or (self.p < 1.0 and random.random() >= self.p):
            return image, target

        selected = random.sample(
            source_objects, min(self.max_pastes, len(source_objects))
        )
        occupied = [polygon for _, polygon in self._polygons_from_target(target)]
        augmented_image = image.copy()
        augmented_target = copy.deepcopy(target)
        pasted = False
        for source in selected:
            pasted |= self._paste(
                augmented_image, augmented_target, source, occupied
            )

        if not pasted:
            return image, target
        return augmented_image, augmented_target

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(p={self.p}, "
            f"iou_threshold={self.iou_threshold}, max_pastes={self.max_pastes}, "
            f"max_trials={self.max_trials}, "
            f"max_cached_images={self.max_cached_images})"
        )
