from __future__ import annotations

import argparse
import csv
from itertools import product
from pathlib import Path

from ultralytics import YOLO


def parse_list(text: str):
    return [float(x) for x in text.split(',') if x.strip()]


def main():
    parser = argparse.ArgumentParser(description='Threshold sweep for stage-2 lesion-only YOLO model.')
    parser.add_argument('--weights', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--imgsz', type=int, default=960)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--device', default='0')
    parser.add_argument('--split', default='val')
    parser.add_argument('--project', default='runs/stage2_lesion')
    parser.add_argument('--name', default='sweep_stage2')
    parser.add_argument('--confs', default='0.10,0.15,0.20,0.25,0.30')
    parser.add_argument('--ious', default='0.45,0.50,0.55,0.60')
    args = parser.parse_args()

    confs = parse_list(args.confs)
    ious = parse_list(args.ious)
    out_dir = Path(args.project) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'sweep.csv'

    model = YOLO(args.weights)
    rows = []
    best = None
    for conf, iou in product(confs, ious):
        print(f'>>> val split={args.split} conf={conf} iou={iou}')
        metrics = model.val(
            data=args.data,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            split=args.split,
            conf=conf,
            iou=iou,
            verbose=False,
            save=False,
            plots=False,
        )
        row = {
            'split': args.split,
            'conf': conf,
            'iou': iou,
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr),
            'map50': float(metrics.box.map50),
            'map50_95': float(metrics.box.map),
            'fitness': float(metrics.box.map),
        }
        rows.append(row)
        if best is None or row['fitness'] > best['fitness']:
            best = row

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f'Saved sweep CSV to: {csv_path}')
    print(f'Best row: {best}')


if __name__ == '__main__':
    main()
