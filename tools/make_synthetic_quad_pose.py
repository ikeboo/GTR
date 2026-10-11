"""Generate a small COCO keypoint dataset of gently warped rectangles."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


KEYPOINT_NAMES = ['top_left', 'top_right', 'bottom_right', 'bottom_left']
SOURCE_CORNERS = np.array([[32, 40], [96, 40], [96, 88], [32, 88]], dtype=np.float32)


def make_split(root, name, count, rng):
    image_dir = root / name
    image_dir.mkdir(parents=True, exist_ok=True)
    images, annotations = [], []
    for index in range(count):
        background = int(rng.integers(20, 65))
        canvas = np.full((128, 128, 3), background, dtype=np.uint8)
        cv2.fillConvexPoly(canvas, SOURCE_CORNERS.astype(np.int32), (175, 210, 245))
        for point, color in zip(SOURCE_CORNERS, [(40, 40, 240), (40, 220, 40),
                                                 (240, 80, 40), (30, 220, 220)]):
            cv2.circle(canvas, tuple(point.astype(int)), 5, color, -1)

        destination = SOURCE_CORNERS + rng.uniform(-7, 7, (4, 2)).astype(np.float32)
        homography = cv2.getPerspectiveTransform(SOURCE_CORNERS, destination)
        warped = cv2.warpPerspective(canvas, homography, (128, 128),
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(background,) * 3)
        corners = cv2.perspectiveTransform(SOURCE_CORNERS[None], homography)[0]
        file_name = f'{index:04d}.png'
        cv2.imwrite(str(image_dir / file_name), warped)

        x0, y0 = corners.min(axis=0)
        x1, y1 = corners.max(axis=0)
        area = float(cv2.contourArea(corners))
        annotations.append({
            'id': index + 1, 'image_id': index + 1, 'category_id': 1,
            'bbox': [float(x0), float(y0), float(x1 - x0), float(y1 - y0)],
            'area': area, 'iscrowd': 0,
            'num_keypoints': 4,
            'keypoints': [value for x, y in corners for value in (float(x), float(y), 2)],
            'segmentation': [corners.reshape(-1).astype(float).tolist()],
        })
        images.append({'id': index + 1, 'file_name': file_name, 'width': 128, 'height': 128})

    data = {
        'info': {'description': 'Synthetic homography quadrilateral keypoints'},
        'licenses': [], 'images': images, 'annotations': annotations,
        'categories': [{'id': 1, 'name': 'quadrilateral', 'supercategory': 'shape',
                        'keypoints': KEYPOINT_NAMES,
                        'skeleton': [[1, 2], [2, 3], [3, 4], [4, 1]]}],
    }
    (root / f'{name}.json').write_text(json.dumps(data, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('dataset/synthetic_quad_pose'))
    parser.add_argument('--train-count', type=int, default=24)
    parser.add_argument('--val-count', type=int, default=8)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    if args.train_count < 1 or args.val_count < 1:
        parser.error('Both split sizes must be positive')
    rng = np.random.default_rng(args.seed)
    make_split(args.output, 'train', args.train_count, rng)
    make_split(args.output, 'val', args.val_count, rng)
    print(f'Created {args.train_count} train and {args.val_count} val images in {args.output}')


if __name__ == '__main__':
    main()
