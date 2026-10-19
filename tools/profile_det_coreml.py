"""Inspect local CoreML device assignment and time repeated patch inference.

The compute plan is an estimate for the host device, not an iPhone benchmark.
Timing excludes model loading, preprocessing, postprocessing and image merging.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import platform
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from coremltools.models.compute_plan import MLComputePlan
from tools.inference.coreml_inf import COMPUTE_UNITS, CoreMLDetector
from tools.inference.coreml_split_inf import CoreMLSplitDetector


def compute_plan_report(model, compute_units):
    print('Loading host compute plan', flush=True)
    plan = MLComputePlan.load_from_path(model.get_compiled_model_path(), COMPUTE_UNITS[compute_units])
    print('Inspecting host compute plan', flush=True)
    counts, costs, by_op, supported, op_costs = Counter(), Counter(), Counter(), Counter(), Counter()
    non_neural_engine = []
    missing_costs = 0
    def visit(block):
        nonlocal missing_costs
        for op in block.operations:
            if op.operator_name == 'const':
                continue
            usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
            cost = plan.get_estimated_cost_for_mlprogram_operation(op)
            device = type(usage.preferred_compute_device).__name__ if usage else 'Unknown'
            counts[device] += 1
            by_op[(device, op.operator_name)] += 1
            if usage:
                for supported_device in usage.supported_compute_devices:
                    supported[type(supported_device).__name__] += 1
                if not any('NeuralEngine' in type(d).__name__ for d in usage.supported_compute_devices):
                    non_neural_engine.append({'op': op.operator_name,
                                              'outputs': [out.name for out in op.outputs],
                                              'preferred_device': device})
            if cost is None:
                missing_costs += 1
            else:
                costs[device] += cost.weight
                op_costs[(device, op.operator_name)] += cost.weight
            for child in op.blocks:
                visit(child)
    if plan.model_structure.program is None:
        raise ValueError('Expected an ML Program')
    for function in plan.model_structure.program.functions.values():
        visit(function.block)
    return {'preferred_device_operations': dict(counts),
            'supported_device_operations': dict(supported),
            'estimated_cost_weight_by_device': dict(costs),
            'operations_without_cost': missing_costs,
            'non_neural_engine_operations': non_neural_engine,
            'operation_counts': [{'device': d, 'op': o, 'count': n,
                                  'estimated_cost_weight': op_costs[(d, o)]}
                                 for (d, o), n in by_op.most_common()],
            'note': 'Host-specific anticipated assignment; cost weights are estimates, not measured time fractions'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--image', type=Path, default=Path('assets/00.jpg'))
    parser.add_argument('--compute-units', choices=COMPUTE_UNITS, default='all')
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--runs', type=int, default=30)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--skip-plan', action='store_true', help='Measure without the expensive compute-plan compilation')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.runs < 1 or args.warmup < 0:
        parser.error('runs must be positive and warmup nonnegative')
    if args.plan_only and args.skip_plan:
        parser.error('plan-only and skip-plan cannot be combined')
    start = time.perf_counter()
    detector_class = CoreMLSplitDetector if args.model.suffix == '.json' else CoreMLDetector
    with detector_class(args.model, compute_units=args.compute_units) as detector:
        print('Model loaded', flush=True)
        report = {'model': str(args.model), 'compute_units': args.compute_units,
                  'host': platform.platform(), 'load_seconds': time.perf_counter() - start,
                  'compute_plan': (None if args.skip_plan else
                                   compute_plan_report(detector.model, args.compute_units))}
        if isinstance(detector, CoreMLSplitDetector):
            report['decoder_compute_units'] = detector.decoder_compute_units
            report['decoder_compute_plan'] = (None if args.skip_plan else
                compute_plan_report(detector.decoder, detector.decoder_compute_units))
        if not args.plan_only:
            image = np.asarray(Image.open(args.image).convert('RGB'))[:, :, ::-1]
            sample = detector.preprocess(image)
            for _ in range(args.warmup):
                detector.predict_raw(sample)
            durations = []
            for _ in range(args.runs):
                start = time.perf_counter()
                detector.predict_raw(sample)
                durations.append(time.perf_counter() - start)
            report['timing'] = {'runs': args.runs, 'warmup': args.warmup,
                                'median_ms': float(np.median(durations) * 1000),
                                'p95_ms': float(np.percentile(durations, 95) * 1000),
                                'total_seconds': sum(durations),
                                'samples_ms': [t * 1000 for t in durations],
                                'note': 'Repeated identical patch, includes Python binding/output copy; excludes preprocessing/postprocessing'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'compute_plan'}, indent=2))
    if report['compute_plan'] is not None:
        print(json.dumps(report['compute_plan']['preferred_device_operations']))


if __name__ == '__main__':
    main()
