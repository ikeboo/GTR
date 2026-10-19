"""GTR detection inference on macOS, without importing PyTorch.

Images are BGR uint8 numpy arrays (OpenCV convention). The ML Program returns
raw logits and normalized cxcywh boxes; sigmoid/top-k and pixel scaling run in
float32 here, matching the detection PostProcessor. No NMS is added.
"""

import argparse
import json
from pathlib import Path
import threading
import time

import coremltools as ct
import numpy as np
from PIL import Image


COMPUTE_UNITS = {
    'cpu_only': ct.ComputeUnit.CPU_ONLY,
    'cpu_and_gpu': ct.ComputeUnit.CPU_AND_GPU,
    'cpu_and_ne': ct.ComputeUnit.CPU_AND_NE,
    'all': ct.ComputeUnit.ALL,
}


class CoreMLDetector:
    def __init__(self, model_path, threshold=0.4, compute_units='cpu_and_gpu'):
        if not 0 <= threshold <= 1:
            raise ValueError('threshold must be between 0 and 1')
        self.threshold = threshold
        self.model = ct.models.MLModel(str(model_path), compute_units=COMPUTE_UNITS[compute_units])
        self.compute_units = compute_units
        metadata = self.model.user_defined_metadata
        self.input_size = tuple(json.loads(metadata['input_size']))
        self.num_classes = int(metadata['num_classes'])
        self.output_class_slots = int(metadata.get('output_class_slots', self.num_classes))
        self.num_top_queries = int(metadata['num_top_queries'])
        self.category_ids = json.loads(metadata.get('category_ids', 'null'))
        self.category_names = json.loads(metadata.get('category_names', 'null'))
        # coremltools 9.0 can release input numpy arrays on a CoreML dispatch
        # queue without the GIL (apple/coremltools#2829). A persistent buffer
        # keeps Python ownership across calls, including the idle reset window.
        self._input_buffer = np.empty((1, 3, *self.input_size), dtype=np.float32)
        self._predict_lock = threading.Lock()

    def preprocess(self, image):
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError('Expected an HWC BGR uint8 image')
        height, width = self.input_size
        rgb = Image.fromarray(image[:, :, ::-1]).resize((width, height), Image.Resampling.BILINEAR)
        value = np.asarray(rgb, dtype=np.float32) / 255.0
        value = (value - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32)
        return np.ascontiguousarray(value.transpose(2, 0, 1)[None])

    def predict_raw(self, images):
        expected_shape = (1, 3, *self.input_size)
        if images.shape != expected_shape:
            raise ValueError(f'Expected input shape {expected_shape}, got {images.shape}')
        with self._predict_lock:
            if self.model is None:
                raise RuntimeError('Detector is closed')
            np.copyto(self._input_buffer, images, casting='same_kind')
            outputs = self.model.predict({'images': self._input_buffer})
            logits = np.array(outputs['pred_logits'], dtype=np.float32, copy=True).reshape(
                1, 300, self.output_class_slots)[..., :self.num_classes].copy()
            boxes = np.array(outputs['pred_boxes'], dtype=np.float32, copy=True).reshape(1, 300, 4)
        if logits.shape != (1, 300, self.num_classes) or boxes.shape != (1, 300, 4):
            raise ValueError(f'Unexpected output shapes: {logits.shape}, {boxes.shape}')
        if not (np.isfinite(logits).all() and np.isfinite(boxes).all()):
            raise RuntimeError('CoreML returned non-finite outputs')
        return {'pred_logits': logits, 'pred_boxes': boxes}

    def close(self):
        """Release CoreML before releasing the persistent numpy input.

        The 9.0 binding can hold its input for about one second after predict.
        Keep the buffer owned by Python while that delayed release completes.
        """
        lock = getattr(self, '_predict_lock', None)
        if lock is None:
            return
        with lock:
            if self.model is not None:
                self.model = None
                time.sleep(1.2)
                self._input_buffer = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()

    def postprocess(self, outputs, original_size, threshold=None):
        """Return labels, xyxy boxes, scores, and query indices in score order.

        original_size is (width, height). Labels retain the checkpoint's class
        indexing; Objects365 uses 366 slots, including the unused slot zero.
        """
        threshold = self.threshold if threshold is None else threshold
        if not 0 <= threshold <= 1:
            raise ValueError('threshold must be between 0 and 1')
        logits = outputs['pred_logits'][0]
        scores = np.empty_like(logits)
        positive = logits >= 0
        scores[positive] = 1 / (1 + np.exp(-logits[positive]))
        exp = np.exp(logits[~positive])
        scores[~positive] = exp / (1 + exp)
        # Stable ties give reproducible results across numpy versions.
        indices = np.argsort(-scores.ravel(), kind='stable')[:self.num_top_queries]
        selected_scores = scores.ravel()[indices]
        keep = selected_scores > threshold
        indices, selected_scores = indices[keep], selected_scores[keep]
        labels = (indices % self.num_classes).astype(np.int32)
        queries = indices // self.num_classes
        boxes = outputs['pred_boxes'][0, queries]
        centers, sizes = boxes[:, :2], boxes[:, 2:]
        boxes = np.concatenate((centers - sizes / 2, centers + sizes / 2), axis=-1)
        boxes *= np.asarray([*original_size, *original_size], dtype=np.float32)
        if self.category_ids is not None:
            labels = np.asarray(self.category_ids, dtype=np.int32)[labels]
        return {'labels': labels, 'boxes': boxes, 'scores': selected_scores,
                'query_indices': queries.astype(np.int32)}

    def __call__(self, image, threshold=None):
        return self.postprocess(self.predict_raw(self.preprocess(image)),
                                (image.shape[1], image.shape[0]), threshold)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--threshold', type=float, default=0.4)
    parser.add_argument('--compute-units', choices=COMPUTE_UNITS, default='cpu_and_gpu')
    parser.add_argument('--output', type=Path, help='Optional JSON result path')
    args = parser.parse_args()
    image = np.asarray(Image.open(args.image).convert('RGB'))[:, :, ::-1]
    with CoreMLDetector(args.model, args.threshold, args.compute_units) as detector:
        result = {name: value.tolist() for name, value in detector(image).items()}
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + '\n')
    else:
        print(serialized)


if __name__ == '__main__':
    main()
