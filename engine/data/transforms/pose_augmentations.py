"""Pose image augmentations adapted from repos/ecpose.

These operate on (image, pixel-space target) pairs before PoseNormalize.
"""

import numbers
import random

import cv2
import numpy as np
import PIL
import torch
import torch.nn.functional as torch_F
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
from PIL import Image
from torchvision import transforms

from ...core import register


@register()
class RandomRotation(object):
    """Rotate an image and its annotations without cropping keypoints.

    The angle is sampled uniformly from ``[-degrees, degrees]``. Before the
    rotation, the image is scaled by the largest factor that keeps all four
    rotated image corners inside the original canvas. Consequently, every
    keypoint that starts inside the image remains inside it after rotation.

    Args:
        degrees: Maximum absolute rotation angle in degrees.
        p: Probability of applying the transform.
        fill: Pixel value used for areas outside the transformed image.
    """

    def __init__(self, degrees, p=0.5, fill=0):
        if not isinstance(degrees, numbers.Number) or isinstance(degrees, bool):
            raise TypeError("degrees must be a number")
        if not np.isfinite(degrees) or degrees < 0:
            raise ValueError("degrees must be a finite, non-negative number")
        if not isinstance(p, numbers.Number) or isinstance(p, bool):
            raise TypeError("p must be a number")
        if not 0 <= p <= 1:
            raise ValueError("p must be between 0 and 1")

        self.degrees = float(degrees)
        self.p = float(p)
        self.fill = fill

    @staticmethod
    def _fit_scale(width, height, angle):
        radians = np.deg2rad(angle)
        abs_cos = abs(float(np.cos(radians)))
        abs_sin = abs(float(np.sin(radians)))
        rotated_width = width * abs_cos + height * abs_sin
        rotated_height = width * abs_sin + height * abs_cos
        return min(width / rotated_width, height / rotated_height, 1.0)

    @staticmethod
    def _transform_xy(points, width, height, angle, scale):
        if points.numel() == 0:
            return points.clone()

        dtype = points.dtype if torch.is_floating_point(points) else torch.float32
        points = points.to(dtype=dtype)
        radians = np.deg2rad(angle)
        cos = float(np.cos(radians)) * scale
        sin = float(np.sin(radians)) * scale
        matrix = points.new_tensor(((cos, sin), (-sin, cos)))
        center = points.new_tensor((width / 2.0, height / 2.0))
        transformed = (points - center) @ matrix.T + center
        transformed[..., 0].clamp_(0.0, float(width))
        transformed[..., 1].clamp_(0.0, float(height))
        return transformed

    def _transform_target(self, target, width, height, angle, scale):
        if target is None:
            return None

        target = target.copy()

        if "keypoints" in target:
            keypoints = target["keypoints"]
            if keypoints.shape[-1] < 2:
                raise ValueError("keypoints must have at least x and y coordinates")
            dtype = (
                keypoints.dtype
                if torch.is_floating_point(keypoints)
                else torch.float32
            )
            transformed_keypoints = keypoints.to(dtype=dtype).clone()
            transformed_xy = self._transform_xy(
                transformed_keypoints[..., :2], width, height, angle, scale
            )
            if keypoints.shape[-1] >= 3:
                visible = transformed_keypoints[..., 2] > 0
                transformed_xy = torch.where(
                    visible.unsqueeze(-1), transformed_xy,
                    torch.zeros_like(transformed_xy)
                )
            transformed_keypoints[..., :2] = transformed_xy
            target["keypoints"] = transformed_keypoints

        if "boxes" in target:
            boxes = target["boxes"]
            if boxes.shape[-1] != 4:
                raise ValueError("boxes must use the xyxy format")
            dtype = boxes.dtype if torch.is_floating_point(boxes) else torch.float32
            boxes = boxes.to(dtype=dtype)
            x1, y1, x2, y2 = boxes.unbind(-1)
            corners = torch.stack(
                (
                    torch.stack((x1, y1), dim=-1),
                    torch.stack((x2, y1), dim=-1),
                    torch.stack((x2, y2), dim=-1),
                    torch.stack((x1, y2), dim=-1),
                ),
                dim=-2,
            )
            transformed_corners = self._transform_xy(
                corners, width, height, angle, scale
            )
            minimum = transformed_corners.amin(dim=-2)
            maximum = transformed_corners.amax(dim=-2)
            target["boxes"] = torch.cat((minimum, maximum), dim=-1)

        if "area" in target:
            target["area"] = target["area"] * (scale ** 2)

        if "masks" in target:
            target["masks"] = F.affine(
                target["masks"],
                angle=-angle,
                translate=[0, 0],
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=transforms.InterpolationMode.NEAREST,
                fill=0,
                center=[width / 2.0, height / 2.0],
            )

        return target

    def __call__(self, img, target=None):
        if random.random() >= self.p:
            return img, target

        height, width = F.get_size(img)
        angle = random.uniform(-self.degrees, self.degrees)
        scale = self._fit_scale(width, height, angle)
        rotated_img = F.affine(
            img,
            angle=-angle,
            translate=[0, 0],
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=self.fill,
            center=[width / 2.0, height / 2.0],
        )
        target = self._transform_target(
            target, width, height, angle, scale
        )
        return rotated_img, target

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(degrees={self.degrees}, "
            f"p={self.p}, fill={self.fill})"
        )




@register()
class GaussianNoise(object):
    """Add Gaussian noise to a PIL or tensor image.

    ``mean`` and ``std`` are specified on the normalized ``[0, 1]`` image
    scale. Place this transform before ``Normalize`` in the YAML pipeline.
    """

    def __init__(self, mean=0.0, std=0.05, p=0.5, clip=True):
        if not isinstance(mean, numbers.Number):
            raise TypeError("mean must be a number")
        if not isinstance(std, numbers.Number):
            raise TypeError("std must be a number")
        if std < 0:
            raise ValueError("std must be non-negative")
        if not 0 <= p <= 1:
            raise ValueError("p must be between 0 and 1")

        self.mean = float(mean)
        self.std = float(std)
        self.p = float(p)
        self.clip = bool(clip)

    def _apply_tensor(self, image):
        original_dtype = image.dtype
        is_floating_point = torch.is_floating_point(image)

        if is_floating_point:
            working_image = image
        else:
            max_value = torch.iinfo(original_dtype).max
            working_image = image.to(torch.float32) / max_value

        noise = torch.randn_like(working_image) * self.std + self.mean
        output = working_image + noise
        if self.clip:
            output = output.clamp(0.0, 1.0)

        if not is_floating_point:
            # Integer image formats cannot represent values outside their
            # valid range even when clip=False.
            output = output.clamp(0.0, 1.0)
            output = (output * max_value).round().to(original_dtype)

        return output

    def __call__(self, img, target=None):
        if random.random() >= self.p:
            return img, target

        if isinstance(img, PIL.Image.Image):
            tensor = F.pil_to_tensor(img)
            noisy_tensor = self._apply_tensor(tensor)
            img = transforms.functional.to_pil_image(noisy_tensor, mode=img.mode)
        elif isinstance(img, torch.Tensor):
            img = self._apply_tensor(img)
        else:
            raise TypeError(
                f"GaussianNoise expects a PIL image or tensor, got {type(img)}"
            )

        return img, target

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(mean={self.mean}, std={self.std}, "
            f"p={self.p}, clip={self.clip})"
        )


@register()
class GaussianBlur(object):
    """Apply Gaussian blur to a PIL or tensor image with probability ``p``."""

    def __init__(self, kernel_size=5, sigma=(0.1, 2.0), p=0.5):
        if not 0 <= p <= 1:
            raise ValueError("p must be between 0 and 1")

        self.kernel_size = kernel_size
        self.sigma = sigma
        self.p = float(p)
        # Let torchvision validate kernel_size and sigma consistently with its
        # native augmentation implementation.
        self.transform = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)

    def __call__(self, img, target=None):
        if random.random() < self.p:
            img = self.transform(img)
        return img, target

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(kernel_size={self.kernel_size}, "
            f"sigma={self.sigma}, p={self.p})"
        )


@register()
class JPEGDeter(object):
    """Simulate JPEG degradation with an in-memory encode/decode round trip.

    The transform uses OpenCV's native JPEG codec to keep the per-image cost
    low. For best performance, place it before ``ToTensor`` in the pipeline.
    Floating-point tensor inputs are expected to use the ``[0, 1]`` range.

    Args:
        quality: A fixed JPEG quality or an inclusive ``(min, max)`` range.
        p: Probability of applying the transform.
    """

    def __init__(self, quality=(30, 95), p=0.5):
        self.quality = self._validate_quality(quality)
        if not isinstance(p, numbers.Number):
            raise TypeError("p must be a number")
        if not 0 <= p <= 1:
            raise ValueError("p must be between 0 and 1")
        self.p = float(p)

    @staticmethod
    def _validate_quality(quality):
        if isinstance(quality, numbers.Integral) and not isinstance(quality, bool):
            quality = (int(quality), int(quality))
        elif isinstance(quality, (tuple, list)) and len(quality) == 2:
            if not all(
                isinstance(value, numbers.Integral) and not isinstance(value, bool)
                for value in quality
            ):
                raise TypeError("quality values must be integers")
            quality = (int(quality[0]), int(quality[1]))
        else:
            raise TypeError("quality must be an integer or a pair of integers")

        if not 1 <= quality[0] <= quality[1] <= 100:
            raise ValueError("quality must satisfy 1 <= min <= max <= 100")
        return quality

    @staticmethod
    def _compress(image, quality):
        is_grayscale = image.ndim == 2
        if not is_grayscale:
            # OpenCV's JPEG codec expects BGR while PIL and image tensors use
            # RGB. The conversions preserve realistic JPEG luma/chroma loss.
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        success, encoded = cv2.imencode(
            ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        if not success:
            raise RuntimeError("failed to encode image as JPEG")

        decode_mode = cv2.IMREAD_GRAYSCALE if is_grayscale else cv2.IMREAD_COLOR
        decoded = cv2.imdecode(encoded, decode_mode)
        if decoded is None:
            raise RuntimeError("failed to decode JPEG image")
        if not is_grayscale:
            decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        return decoded

    def _apply_pil(self, image, quality):
        # JPEG has no alpha channel. Non-grayscale inputs are intentionally
        # converted to RGB, matching how training images are normally loaded.
        source = image if image.mode == "L" else image.convert("RGB")
        array = np.ascontiguousarray(np.asarray(source))
        decoded = self._compress(array, quality)
        return Image.fromarray(decoded)

    def _apply_tensor(self, image, quality):
        if image.ndim != 3 or image.shape[0] not in (1, 3):
            raise ValueError(
                "JPEGDeter expects a CHW tensor with one or three channels"
            )

        original_dtype = image.dtype
        if torch.is_floating_point(image):
            uint8_image = (image.detach().clamp(0, 1) * 255).round().to(torch.uint8)
        elif original_dtype == torch.uint8:
            uint8_image = image.detach()
        else:
            raise TypeError("JPEGDeter tensor input must be floating point or uint8")

        array = uint8_image.cpu().permute(1, 2, 0).numpy()
        if array.shape[2] == 1:
            array = array[:, :, 0]
        decoded = self._compress(np.ascontiguousarray(array), quality)
        if decoded.ndim == 2:
            decoded = decoded[:, :, None]

        output = torch.from_numpy(decoded.copy()).permute(2, 0, 1)
        if torch.is_floating_point(image):
            output = output.to(dtype=original_dtype).div_(255)
        return output.to(device=image.device)

    def __call__(self, img, target=None):
        if random.random() >= self.p:
            return img, target

        quality = random.randint(*self.quality)
        if isinstance(img, PIL.Image.Image):
            img = self._apply_pil(img, quality)
        elif isinstance(img, torch.Tensor):
            img = self._apply_tensor(img, quality)
        else:
            raise TypeError(
                f"JPEGDeter expects a PIL image or tensor, got {type(img)}"
            )
        return img, target

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(quality={self.quality}, p={self.p})"
        )


@register()
class MotionBlur(object):
    """Apply inexpensive linear motion blur with a random length and angle.

    A single normalized line kernel and one depth-wise convolution are used,
    which approximates constant camera/object motion without generating or
    blending multiple frames.

    Args:
        kernel_size: A fixed odd length or an inclusive ``(min, max)`` range.
            When a range is given, one of the odd lengths in it is sampled.
        angle: A fixed angle in degrees or a ``(min, max)`` range.
        p: Probability of applying the transform.
    """

    def __init__(self, kernel_size=(3, 9), angle=(-180.0, 180.0), p=0.5):
        self.kernel_size = self._validate_kernel_size(kernel_size)
        self.angle = self._validate_angle(angle)
        if not isinstance(p, numbers.Number):
            raise TypeError("p must be a number")
        if not 0 <= p <= 1:
            raise ValueError("p must be between 0 and 1")
        self.p = float(p)

    @staticmethod
    def _validate_kernel_size(kernel_size):
        if (
            isinstance(kernel_size, numbers.Integral)
            and not isinstance(kernel_size, bool)
        ):
            kernel_size = int(kernel_size)
            if kernel_size < 1 or kernel_size % 2 == 0:
                raise ValueError("kernel_size must be a positive odd integer")
            return kernel_size

        if isinstance(kernel_size, (tuple, list)) and len(kernel_size) == 2:
            if not all(
                isinstance(value, numbers.Integral) and not isinstance(value, bool)
                for value in kernel_size
            ):
                raise TypeError("kernel_size range values must be integers")
            kernel_size = (int(kernel_size[0]), int(kernel_size[1]))
            if kernel_size[0] < 1 or kernel_size[0] > kernel_size[1]:
                raise ValueError("kernel_size must satisfy 1 <= min <= max")
            first_odd = kernel_size[0] | 1
            if first_odd > kernel_size[1]:
                raise ValueError("kernel_size range must contain an odd integer")
            return kernel_size

        raise TypeError("kernel_size must be an odd integer or a pair of integers")

    @staticmethod
    def _validate_angle(angle):
        if isinstance(angle, numbers.Number):
            angle = float(angle)
            if not np.isfinite(angle):
                raise ValueError("angle must be finite")
            return angle

        if isinstance(angle, (tuple, list)) and len(angle) == 2:
            if not all(isinstance(value, numbers.Number) for value in angle):
                raise TypeError("angle range values must be numbers")
            angle = (float(angle[0]), float(angle[1]))
            if not all(np.isfinite(value) for value in angle):
                raise ValueError("angle values must be finite")
            if angle[0] > angle[1]:
                raise ValueError("angle must satisfy min <= max")
            return angle

        raise TypeError("angle must be a number or a pair of numbers")

    def _sample_kernel_size(self):
        if isinstance(self.kernel_size, int):
            return self.kernel_size
        minimum, maximum = self.kernel_size
        first_odd = minimum | 1
        return random.randrange(first_odd, maximum + 1, 2)

    def _sample_angle(self):
        if isinstance(self.angle, float):
            return self.angle
        return random.uniform(*self.angle)

    @staticmethod
    def _make_kernel(kernel_size, angle):
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        center = (kernel_size - 1) / 2.0
        radians = np.deg2rad(angle)
        dx = np.cos(radians) * center
        dy = np.sin(radians) * center
        start = (round(center - dx), round(center - dy))
        end = (round(center + dx), round(center + dy))
        cv2.line(kernel, start, end, color=1.0, thickness=1)
        kernel /= kernel.sum()
        return kernel

    @staticmethod
    def _apply_pil(image, kernel):
        source = image if image.mode in ("L", "RGB") else image.convert("RGB")
        array = np.ascontiguousarray(np.asarray(source))
        output = cv2.filter2D(
            array, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_REFLECT_101
        )
        return Image.fromarray(output)

    @staticmethod
    def _apply_tensor(image, kernel):
        if image.ndim != 3:
            raise ValueError("MotionBlur expects a CHW tensor")

        original_dtype = image.dtype
        if torch.is_floating_point(image):
            working = image
            if original_dtype not in (torch.float32, torch.float64):
                working = working.float()
        else:
            working = image.float()

        channels = working.shape[0]
        weight = torch.as_tensor(kernel, dtype=working.dtype, device=working.device)
        weight = weight.expand(channels, 1, *weight.shape)
        padding = kernel.shape[0] // 2
        padding_mode = (
            "reflect"
            if working.shape[-2] > padding and working.shape[-1] > padding
            else "replicate"
        )
        padded = torch_F.pad(
            working.unsqueeze(0), (padding, padding, padding, padding),
            mode=padding_mode
        )
        output = torch_F.conv2d(padded, weight, groups=channels).squeeze(0)

        if torch.is_floating_point(image):
            return output.to(dtype=original_dtype)
        dtype_info = torch.iinfo(original_dtype)
        return output.round().clamp(dtype_info.min, dtype_info.max).to(original_dtype)

    def __call__(self, img, target=None):
        if random.random() >= self.p:
            return img, target

        kernel = self._make_kernel(
            self._sample_kernel_size(), self._sample_angle()
        )
        if isinstance(img, PIL.Image.Image):
            img = self._apply_pil(img, kernel)
        elif isinstance(img, torch.Tensor):
            img = self._apply_tensor(img, kernel)
        else:
            raise TypeError(
                f"MotionBlur expects a PIL image or tensor, got {type(img)}"
            )
        return img, target

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(kernel_size={self.kernel_size}, "
            f"angle={self.angle}, p={self.p})"
        )
