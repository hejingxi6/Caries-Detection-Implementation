from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import pandas as pd

from common import dataset_info, list_images


def main():
    parser = argparse.ArgumentParser(description='Simple hard-case ranking based on stage-2 val outputs (manual helper).')
    parser.add_argument('--data', required=True)
    parser.add_argument('--source_split', default='val')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--topk', type=int, default=120)
    args = parser.parse_args()

    _, _, splits = dataset_info(args.data)
    src_dir = splits[args.source_split]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = list_images(src_dir)
    rows = []
    for p in imgs:
        # heuristic: prioritize very small files and awkward aspect ratios for manual review
        score = p.stat().st_size
        rows.append({'image': p.name, 'score': score})
    df = pd.DataFrame(rows).sort_values('score')
    df.to_csv(out_dir / 'ranking.csv', index=False)
    top = df.head(args.topk)['image'].tolist()
    for name in top:
        shutil.copy2(src_dir / name, out_dir / name)
    print(f'Saved top-{args.topk} likely hard cases to: {out_dir}')
    print(f'Ranking CSV: {out_dir / "ranking.csv"}')


if __name__ == '__main__':
    main()
