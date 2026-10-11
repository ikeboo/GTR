"""Shared OKS scales for pose training and COCO evaluation."""

import numpy as np


def get_keypoint_sigmas(num_keypoints, sigmas=None):
    """Return per-point OKS scales, preserving standard human layouts."""
    if not isinstance(num_keypoints, int) or num_keypoints < 1:
        raise ValueError('num_keypoints must be a positive integer')
    if sigmas is not None:
        values = np.asarray(sigmas, dtype=np.float32)
        if values.shape != (num_keypoints,) or not np.all(np.isfinite(values)) or np.any(values <= 0):
            raise ValueError(f'Expected {num_keypoints} positive, finite keypoint sigmas')
        return values
    if num_keypoints == 17:
        return np.array([
            .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07,
            1.07, .87, .87, .89, .89
        ], dtype=np.float32) / 10.0
    if num_keypoints == 14:
        return np.array([
            .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89,
            .79, .79
        ], dtype=np.float32) / 10.0
    if num_keypoints == 3:
        return np.array([1.07, 1.07, 0.67], dtype=np.float32) / 10.0
    return np.full(num_keypoints, 0.05, dtype=np.float32)
