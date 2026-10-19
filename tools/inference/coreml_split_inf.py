"""GTR split inference: FP16 features on CPU/ANE, FP32 decoder on CPU/GPU."""
import argparse
import json
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import coremltools as ct
import numpy as np
from PIL import Image
from tools.inference.coreml_inf import COMPUTE_UNITS, CoreMLDetector


class CoreMLSplitDetector(CoreMLDetector):
    def __init__(self, manifest_path, threshold=0.4, compute_units='cpu_and_ne'):
        if not 0 <= threshold <= 1:
            raise ValueError('threshold must be between 0 and 1')
        path = Path(manifest_path)
        metadata = json.loads(path.read_text())
        self.threshold = threshold
        self.input_size = tuple(metadata['input_size'])
        self.num_classes = metadata['num_classes']
        self.num_top_queries = metadata['num_top_queries']
        self.category_ids = metadata.get('category_ids')
        self.category_names = metadata.get('category_names')
        self.compute_units = compute_units
        self.decoder_compute_units = 'cpu_and_gpu'
        self._predict_lock = threading.Lock()
        self._input_buffer = np.empty((1, 3, *self.input_size), np.float32)
        # coremltools 9.0 widens float16 numpy inputs to float32 and mutates
        # the input dictionary. Allocate that bridge explicitly so the same
        # buffers remain owned by Python across every decoder call. Swift can
        # pass the float16 MLFeatureValue outputs directly without this bridge.
        self._feature_buffers = {f'level{i}': np.empty((1, int(np.prod(shape))), np.float32)
                                 for i, shape in enumerate(metadata['feature_shapes'])}
        self.model = self.decoder = None
        self.model = ct.models.MLModel(str(path.parent / metadata['features_model']),
                                      compute_units=COMPUTE_UNITS[compute_units])
        self.decoder = ct.models.MLModel(str(path.parent / metadata['decoder_model']),
                                        compute_units=ct.ComputeUnit.CPU_AND_GPU)

    def predict_raw(self, images):
        expected_shape = (1, 3, *self.input_size)
        if images.shape != expected_shape:
            raise ValueError(f'Expected input shape {expected_shape}, got {images.shape}')
        with self._predict_lock:
            if self.model is None or self.decoder is None:
                raise RuntimeError('Detector is closed')
            np.copyto(self._input_buffer, images, casting='same_kind')
            features = self.model.predict({'images': self._input_buffer})
            # Persistent buffers avoid the coremltools 9.0 input-lifetime bug.
            # Feature outputs are FP16; widening for the Python binding is exact.
            for name, buffer in self._feature_buffers.items():
                np.copyto(buffer, features[name])
            result = self.decoder.predict(self._feature_buffers)
            outputs = {'pred_logits': np.array(result['pred_logits'], np.float32, copy=True).reshape(
                1, 300, self.num_classes),
                       'pred_boxes': np.array(result['pred_boxes'], np.float32, copy=True).reshape(1, 300, 4)}
        if not all(np.isfinite(value).all() for value in outputs.values()):
            raise RuntimeError('CoreML returned non-finite outputs')
        return outputs

    def close(self):
        lock = getattr(self, '_predict_lock', None)
        if lock is None:
            return
        with lock:
            if self.model is not None or self.decoder is not None:
                self.model = self.decoder = None
                time.sleep(1.2)
                self._input_buffer = None
                self._feature_buffers = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('weights/gtr_s_obj365_ane_split.json'))
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--threshold', type=float, default=0.4)
    parser.add_argument('--compute-units', choices=COMPUTE_UNITS, default='cpu_and_ne',
                        help='Features backend; decoder always uses cpu_and_gpu')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    image = np.asarray(Image.open(args.image).convert('RGB'))[:, :, ::-1]
    with CoreMLSplitDetector(args.model, args.threshold, args.compute_units) as detector:
        result = {name: value.tolist() for name, value in detector(image).items()}
    serialized = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(serialized + '\n')
    else:
        print(serialized)


if __name__ == '__main__':
    main()
