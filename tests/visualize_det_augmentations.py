"""Save 20 training augmentation previews with ground-truth bounding boxes.

Run from the repository root:
    .venv/bin/python tests/visualize_det_augmentations.py

Each PNG compares the original image (left) with the configured training
augmentation (right). Normalization and normalized boxes are undone only for
display. No detector or model weights are needed.
By default, sample images with boxes; --include-empty also samples negatives.
"""

import argparse
import copy
import json
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision.ops import box_convert

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.core import YAMLConfig, create


COLORS = [(255, 80, 80), (60, 220, 120), (80, 160, 255), (255, 200, 50)]


def restore_image(tensor, ops):
    """Undo the configured Normalize operations on the actual pipeline output."""
    tensor = tensor.detach().cpu().as_subclass(torch.Tensor).clone()
    for op in reversed(ops):
        if op['type'] == 'Normalize':
            mean = tensor.new_tensor(op['mean'])[:, None, None]
            std = tensor.new_tensor(op['std'])[:, None, None]
            tensor = tensor * std + mean
    pixels = tensor.clamp(0, 1).mul(255).round().to(torch.uint8)
    return Image.fromarray(pixels.permute(1, 2, 0).numpy())


def restore_boxes(boxes, ops, width, height):
    """Convert the configured final boxes back to pixel-space XYXY for drawing."""
    boxes = boxes.detach().cpu().as_subclass(torch.Tensor).clone()
    fmt = 'xyxy'
    normalized = False
    for op in ops:
        if op['type'] == 'ConvertBoxes':
            fmt = op.get('fmt') or fmt
            normalized = op.get('normalize', False)
    if normalized:
        boxes *= boxes.new_tensor([width, height, width, height])
    return box_convert(boxes, in_fmt=fmt.lower(), out_fmt='xyxy')


def draw_boxes(image, boxes, labels, names):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for box, label in zip(boxes.tolist(), labels.tolist()):
        label = int(label)
        color = COLORS[label % len(COLORS)]
        # Box edges may equal width/height; PIL's last drawable pixel is size-1.
        x1, y1, x2, y2 = box
        x1, x2 = (min(max(x, 0), image.width - 1) for x in (x1, x2))
        y1, y2 = (min(max(y, 0), image.height - 1) for y in (y1, y2))
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        text = f'{names.get(label, str(label))} (id={label})'
        text_width = draw.textbbox((0, 0), text, font=font)[2]
        text_height = draw.textbbox((0, 0), text, font=font)[3]
        text_x = min(x1, max(0, image.width - text_width - 6))
        text_y = max(0, y1 - text_height - 6)
        draw.rectangle((text_x, text_y, text_x + text_width + 6,
                        text_y + text_height + 4), fill=color)
        draw.text((text_x + 3, text_y + 1), text, fill='black', font=font)
    return image


def save_previews(config, output_dir, num_images=20, seed=42, epoch=0, include_empty=False):
    if num_images < 1 or epoch < 0:
        raise ValueError('num_images must be positive and epoch must be non-negative.')
    cfg = YAMLConfig(str(config))
    if cfg.yaml_cfg['task'] != 'detection':
        raise ValueError('This preview requires a detection configuration.')
    global_cfg = cfg.global_cfg
    global_cfg['preview_dataset'] = copy.deepcopy(cfg.yaml_cfg['train_dataloader']['dataset'])
    dataset = create('preview_dataset', global_cfg)
    dataset.set_epoch(epoch)
    if len(dataset) == 0:
        raise ValueError('The training dataset is empty.')
    ops = cfg.yaml_cfg['train_dataloader']['dataset']['transforms']['ops']
    names = dataset.category2name
    if dataset.remap_mscoco_category:
        from engine.data.dataset.coco_dataset import mscoco_label2category
        names = {label: names[category] for label, category in mscoco_label2category.items()
                 if category in names}

    # Favor annotated images for checking box alignment. An augmentation that
    # clips every object out of the image is still kept in the output.
    candidates = list(range(len(dataset)))
    if not include_empty:
        candidates = [index for index, image_id in enumerate(dataset.ids)
                      if any(not ann.get('iscrowd', 0)
                             and ann['bbox'][2] > 0 and ann['bbox'][3] > 0
                             for ann in dataset.coco.imgToAnns.get(image_id, []))]
    if not candidates:
        raise ValueError('No images with boxes found; use --include-empty to include negatives.')
    sampler = random.Random(seed)
    indices = sampler.sample(candidates, min(num_images, len(candidates)))
    if len(indices) < num_images:
        indices += sampler.choices(candidates, k=num_images - len(indices))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for position, index in enumerate(indices):
        sample_seed = seed + position
        random.seed(sample_seed)
        np.random.seed(sample_seed)
        torch.manual_seed(sample_seed)
        original, original_target = dataset.load_item(index)
        # Keep original annotations intact even if an op changes its target in place.
        tensor, target, _ = dataset._transforms(original.copy(), copy.deepcopy(original_target), dataset)
        augmented = restore_image(tensor, ops)
        boxes = restore_boxes(target['boxes'], ops, *augmented.size)
        if not torch.isfinite(boxes).all() or len(boxes) != len(target['labels']):
            raise ValueError(f'Invalid augmented annotations for dataset index {index}.')
        original_boxes = original_target['boxes'].as_subclass(torch.Tensor).clone()
        original_boxes *= original_boxes.new_tensor([augmented.width / original.width,
                                                     augmented.height / original.height] * 2)
        left = draw_boxes(original.resize(augmented.size), original_boxes,
                          original_target['labels'], names)
        right = draw_boxes(augmented, boxes, target['labels'], names)
        image_id = int(target['image_id'].item())
        source_name = dataset.coco.imgs[image_id]['file_name']
        comparison = Image.new('RGB', (left.width + right.width, left.height + 64), '#202020')
        comparison.paste(left, (0, 64))
        comparison.paste(right, (left.width, 64))
        draw = ImageDraw.Draw(comparison)
        draw.text((12, 8), f'Original: {len(original_boxes)} boxes', fill='white')
        draw.text((left.width + 12, 8),
                  f'Augmented: {len(boxes)} boxes | epoch={epoch} seed={sample_seed}', fill='white')
        draw.text((12, 32), f'image_id={image_id} | {source_name}', fill='white')
        filename = f'augmentation_{position + 1:02d}.png'
        comparison.save(output_dir / filename)
        records.append({
            'output': filename, 'source': source_name, 'image_id': image_id,
            'dataset_index': index, 'seed': sample_seed,
            'original_box_count': len(original_boxes), 'augmented_box_count': len(boxes),
            'augmented_boxes_xyxy': boxes.tolist(), 'labels': target['labels'].tolist(),
        })
        print(f'[{position + 1:02d}/{num_images}] {output_dir / filename}')
    manifest = {'config': str(Path(config).resolve()), 'epoch': epoch, 'seed': seed,
                'include_empty': include_empty,
                'transforms': cfg.yaml_cfg['train_dataloader']['dataset']['transforms'],
                'images': records}
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-c', '--config', type=Path, default=ROOT / 'configs/det/gtr_s_custom.yml')
    parser.add_argument('-o', '--output-dir', type=Path, default=ROOT / 'outputs/det_augmentation_preview')
    parser.add_argument('-n', '--num-images', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--include-empty', action='store_true',
                        help='Also sample source images without annotated boxes.')
    parser.add_argument('--epoch', type=int, default=0,
                        help='Training epoch (0-based); epoch >=58 disables strong augmentation in the custom config.')
    args = parser.parse_args()
    save_previews(args.config, args.output_dir, args.num_images, args.seed, args.epoch, args.include_empty)


if __name__ == '__main__':
    main()
