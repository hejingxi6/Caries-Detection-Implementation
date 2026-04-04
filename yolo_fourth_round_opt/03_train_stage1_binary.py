from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from tiny_unet import TinyUNet


class BinarySegDataset(Dataset):
    def __init__(self, root: str | Path, split: str, image_size: int, mean: Sequence[float] | None = None, std: Sequence[float] | None = None):
        self.root = Path(root)
        self.image_paths = sorted((self.root / split / 'images').glob('*'))
        self.mask_paths = {p.stem: p for p in (self.root / split / 'masks').glob('*.png')}
        self.image_size = image_size
        self.mean = np.asarray(mean if mean else [0.5, 0.5, 0.5], dtype=np.float32)
        self.std = np.asarray(std if std else [0.5, 0.5, 0.5], dtype=np.float32)
        self.train = split == 'train'

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[img_path.stem]
        img = cv2.cvtColor(cv2.imread(str(img_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        if self.train:
            if random.random() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1])
                mask = np.ascontiguousarray(mask[:, ::-1])
            if random.random() < 0.2:
                img = np.clip(img.astype(np.float32) * random.uniform(0.9, 1.1), 0, 255).astype(np.uint8)

        img = img.astype(np.float32) / 255.0
        img = (img - self.mean.reshape(1, 1, 3)) / np.clip(self.std.reshape(1, 1, 3), 1e-6, None)
        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        img = np.transpose(img, (2, 0, 1))
        mask = (mask > 127).astype(np.float32)[None, ...]
        return torch.from_numpy(img), torch.from_numpy(mask)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def dice_bce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
    targets = torch.nan_to_num(targets, nan=0.0, posinf=1.0, neginf=0.0)
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets)
    probs = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
    smooth = 1e-6
    inter = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = 1 - ((2 * inter + smooth) / (union + smooth)).mean()
    loss = bce + dice
    if not torch.isfinite(loss):
        loss = torch.zeros((), device=logits.device, dtype=logits.dtype, requires_grad=True)
    return loss


def eval_metrics(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float]:
    logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    smooth = 1e-6
    inter = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = ((2 * inter + smooth) / (union + smooth)).mean().item()
    union_iou = (preds + targets - preds * targets).sum(dim=(1, 2, 3))
    iou = ((inter + smooth) / (union_iou + smooth)).mean().item()
    return dice, iou


def read_norm_json(path: str | None):
    if not path:
        return [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # be tolerant to different key names
    mean = data.get('train_mean_rgb', data.get('mean', [0.5, 0.5, 0.5]))
    std = data.get('train_std_rgb', data.get('std', [0.5, 0.5, 0.5]))
    return mean, std


def train_once(args, batch_size: int) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mean, std = read_norm_json(args.norm_json)

    train_ds = BinarySegDataset(args.dataset_root, 'train', args.image_size, mean, std)
    val_ds = BinarySegDataset(args.dataset_root, 'val', args.image_size, mean, std)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=max(1, batch_size), shuffle=False, num_workers=args.workers, pin_memory=True)

    model = TinyUNet(base=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir = Path(args.project) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    best_iou = -1.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        skipped_batches = 0
        for imgs, masks in tqdm(train_loader, desc=f'train {epoch}/{args.epochs}', leave=False):
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            if not torch.isfinite(imgs).all() or not torch.isfinite(masks).all():
                skipped_batches += 1
                continue
            optimizer.zero_grad(set_to_none=True)
            logits = model(imgs)
            if not torch.isfinite(logits).all():
                skipped_batches += 1
                continue
            loss = dice_bce_loss(logits, masks)
            if not torch.isfinite(loss):
                skipped_batches += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.item()))

        scheduler.step()

        model.eval()
        val_losses, dices, ious = [], [], []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                logits = model(imgs)
                loss = dice_bce_loss(logits, masks)
                if not torch.isfinite(loss):
                    continue
                dice, iou = eval_metrics(logits, masks)
                val_losses.append(float(loss.item()))
                dices.append(float(dice))
                ious.append(float(iou))

        mean_train = float(np.mean(train_losses)) if train_losses else 999.0
        mean_val = float(np.mean(val_losses)) if val_losses else 999.0
        mean_dice = float(np.mean(dices)) if dices else 0.0
        mean_iou = float(np.mean(ious)) if ious else 0.0
        print(f'Epoch {epoch:03d} | train_loss={mean_train:.4f} | val_loss={mean_val:.4f} | val_dice={mean_dice:.4f} | val_iou={mean_iou:.4f} | skipped={skipped_batches}')

        state = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'args': vars(args),
            'batch_size': batch_size,
            'mean': mean,
            'std': std,
            'val_dice': mean_dice,
            'val_iou': mean_iou,
        }
        torch.save(state, out_dir / 'last_stage1.pth')
        if math.isfinite(mean_iou) and mean_iou > best_iou:
            best_iou = mean_iou
            patience_counter = 0
            torch.save(state, out_dir / 'best_stage1.pth')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f'Early stopping triggered at epoch {epoch}.')
                break

    print(f'Finished. Best stage-1 checkpoint: {out_dir / "best_stage1.pth"}')


def main():
    parser = argparse.ArgumentParser(description='Stage-1 binary segmentation on box-rasterized masks.')
    parser.add_argument('--dataset_root', required=True)
    parser.add_argument('--project', default='runs/stage1_binary')
    parser.add_argument('--name', default='y9000p_stage1_binary_unet')
    parser.add_argument('--image_size', type=int, default=640)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--fallback_batch', type=int, default=2)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--base_channels', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--norm_json', default='')
    args = parser.parse_args()

    set_seed(args.seed)
    try:
        print(f'>>> Trying stage-1 batch={args.batch}')
        train_once(args, args.batch)
    except RuntimeError as e:
        msg = str(e).lower()
        if 'out of memory' in msg and args.fallback_batch < args.batch:
            print('[WARN] CUDA OOM detected in stage-1. Retrying with smaller batch...')
            torch.cuda.empty_cache()
            train_once(args, args.fallback_batch)
        else:
            raise


if __name__ == '__main__':
    main()
