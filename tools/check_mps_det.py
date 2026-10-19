"""Check custom detection training, EMA, COCO evaluation and checkpoints on MPS.

Uses a few real samples at the configured resolution, writes only to a separate
smoke-check directory, and never starts the full training run.
"""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from engine.core import YAMLConfig
from engine.optim.lr_scheduler import FlatCosineLRScheduler
from engine.solver.det_engine import evaluate, train_one_epoch
from engine.solver.det_solver import DetSolver


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--config', default='configs/det/gtr_s_custom_mps.yml')
    parser.add_argument('-t', '--tuning', help='Optional detection checkpoint to fine-tune from.')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--train-steps', type=int, default=2)
    parser.add_argument('--val-images', type=int, default=8)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/mps_det_check'))
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        parser.error('Apple MPS is unavailable on this machine.')
    if min(args.batch_size, args.train_steps, args.val_images) < 1 or args.workers < 0:
        parser.error('Batch size / steps / val images must be positive; workers must be non-negative.')

    # A short check uses one optimizer update per batch, independent of the
    # accumulation selected for the full training run.
    os.environ['GTR_GRAD_ACCUM_STEPS'] = '1'
    torch.manual_seed(42)
    cfg = YAMLConfig(args.config, device='mps', use_amp=False, num_workers=args.workers,
                     tuning=args.tuning, output_dir=str(args.output_dir))
    if cfg.task != 'detection':
        parser.error('This check requires a detection configuration.')
    for name in ('train_dataloader', 'val_dataloader'):
        cfg.yaml_cfg[name]['total_batch_size'] = args.batch_size
        cfg.yaml_cfg[name]['num_workers'] = args.workers
    if args.tuning:
        cfg.yaml_cfg['ViTAdapterSpatialSwiGLU']['skip_weights_warning'] = True

    solver = DetSolver(cfg)
    solver.train()
    try:
        train = solver.train_dataloader
        val = solver.val_dataloader
        train.dataset.ids = train.dataset.ids[:args.batch_size * args.train_steps]
        val.dataset.ids = val.dataset.ids[:args.val_images]
        if len(train) == 0:
            parser.error('Not enough training images for one batch.')
        train.set_epoch(0)
        scheduler = FlatCosineLRScheduler(
            solver.optimizer, cfg.lr_gamma, len(train), cfg.epochs,
            warmup_iter=min(cfg.warmup_iter, 3 * len(train)),
            flat_epochs=cfg.flat_epoch if cfg.flat_epoch is not None else train.dataset._transforms.mosaic_epoch,
            no_aug_epochs=cfg.epochs - train.dataset._transforms.stop_epoch,
        )
        tracked = next(parameter for parameter in solver.model.parameters() if parameter.requires_grad)
        before = tracked.detach().cpu().clone()
        started = time.perf_counter()
        training = train_one_epoch(
            True, scheduler, solver.model, solver.criterion, train, solver.optimizer,
            solver.device, epoch=0, max_norm=cfg.clip_max_norm, print_freq=1,
            ema=solver.ema, scaler=solver.scaler,
        )
        torch.mps.synchronize()
        train_seconds = time.perf_counter() - started
        if not math.isfinite(training['loss']):
            raise RuntimeError('Non-finite training loss.')
        for name, parameter in solver.model.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all().item():
                raise RuntimeError(f'Non-finite gradient: {name}')
        changed = not torch.equal(before, tracked.detach().cpu())
        if len(train) > 1 and not changed:
            raise RuntimeError('Model parameters did not change after optimizer updates.')
        module = solver.ema.module if solver.ema else solver.model
        evaluation, _ = evaluate(module, solver.criterion, solver.postprocessor,
                                 val, solver.evaluator, solver.device)
        solver.last_epoch = 0
        checkpoint = args.output_dir / 'smoke.pth'
        torch.save(solver.state_dict(), checkpoint)
        solver.load_state_dict(solver._load_checkpoint(str(checkpoint)))
        report = {
            'device': str(solver.device), 'torch': torch.__version__, 'config': args.config,
            'tuning': args.tuning, 'batch_size': args.batch_size,
            'train_images': len(train.dataset), 'train_steps': len(train),
            'train_seconds': train_seconds, 'train_loss': training['loss'],
            'finite_gradients': True, 'parameters_updated': changed,
            'ema_updates': solver.ema.updates if solver.ema else None,
            'val_images': len(val.dataset), 'evaluation': evaluation,
            'checkpoint_reloaded': str(checkpoint),
            'mps_driver_memory_gb': torch.mps.driver_allocated_memory() / 1e9,
        }
        (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2), flush=True)
    finally:
        if solver.writer:
            solver.writer.close()


if __name__ == '__main__':
    main()
