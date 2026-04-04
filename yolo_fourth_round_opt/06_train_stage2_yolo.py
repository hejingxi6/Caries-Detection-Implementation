from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def run_train(model, args, hyp, batch_value):
    return model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=batch_value,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        pretrained=args.pretrained,
        optimizer=args.optimizer,
        patience=args.patience,
        project=args.project,
        name=args.name,
        exist_ok=True,
        close_mosaic=args.close_mosaic,
        amp=True,
        cache='disk',
        cos_lr=True,
        multi_scale=False,
        overlap_mask=False,
        rect=False,
        val=True,
        save=True,
        plots=True,
        verbose=True,
        save_period=10,
        **hyp,
    )


def main():
    parser = argparse.ArgumentParser(description='Aggressive stage-2 YOLO training on ROI-cropped lesion-only data.')
    parser.add_argument('--data', required=True)
    parser.add_argument('--weights', default='yolov8m.pt')
    parser.add_argument('--project', default='runs/stage2_lesion')
    parser.add_argument('--name', default='y9000p_stage2_yolov8m_roi960_e180')
    parser.add_argument('--imgsz', type=int, default=960)
    parser.add_argument('--epochs', type=int, default=180)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--fallback_batch', type=int, default=1)
    parser.add_argument('--device', default='0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--optimizer', default='AdamW')
    parser.add_argument('--lr0', type=float, default=0.00035)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--warmup_epochs', type=float, default=3.0)
    parser.add_argument('--box', type=float, default=8.0)
    parser.add_argument('--cls', type=float, default=2.5)
    parser.add_argument('--dfl', type=float, default=1.5)
    parser.add_argument('--close_mosaic', type=int, default=20)
    parser.add_argument('--patience', type=int, default=40)
    parser.add_argument('--pretrained', action='store_true')
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.project) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    hyp = {
        'lr0': args.lr0,
        'lrf': args.lrf,
        'weight_decay': args.weight_decay,
        'warmup_epochs': args.warmup_epochs,
        'box': args.box,
        'cls': args.cls,
        'dfl': args.dfl,
        'hsv_h': 0.012,
        'hsv_s': 0.55,
        'hsv_v': 0.30,
        'degrees': 1.5,
        'translate': 0.03,
        'scale': 0.12,
        'shear': 0.3,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 0.60,
        'mixup': 0.05,
        'copy_paste': 0.10,
        'erasing': 0.10,
    }
    with open(out_dir / 'run_config.json', 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2)
    with open(out_dir / 'augmentation_config.json', 'w', encoding='utf-8') as f:
        json.dump(hyp, f, indent=2)

    model = YOLO(args.weights)
    try:
        print(f'>>> Trying stage-2 batch={args.batch}')
        run_train(model, args, hyp, args.batch)
    except RuntimeError as e:
        msg = str(e).lower()
        if 'out of memory' in msg and args.fallback_batch < args.batch:
            print('[WARN] CUDA OOM detected in stage-2. Retrying with smaller batch...')
            torch.cuda.empty_cache()
            model = YOLO(args.weights)
            run_train(model, args, hyp, args.fallback_batch)
        else:
            raise

    best_weight = Path(args.project) / args.name / 'weights' / 'best.pt'
    print(f'Training finished. Best weight: {best_weight}')


if __name__ == '__main__':
    main()
