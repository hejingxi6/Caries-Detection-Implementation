from __future__ import annotations

import argparse
from pathlib import Path

from common import compute_rgb_mean_std, dataset_info, list_images, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description='Compute train-set global RGB mean/std for later normalization.')
    parser.add_argument('--data', required=True, help='Base YOLO dataset yaml')
    parser.add_argument('--out_json', default='norm_stats.json')
    args = parser.parse_args()

    _, _, splits = dataset_info(args.data)
    train_dir = splits.get('train')
    if train_dir is None:
        raise ValueError('train split not found in yaml')

    images = list_images(train_dir)
    print(f'Found {len(images)} training images in {train_dir}')
    mean, std = compute_rgb_mean_std(images)
    out = {'train_mean_rgb': mean, 'train_std_rgb': std, 'num_images': len(images)}
    save_json(args.out_json, out)
    print(f'Saved norm stats to {Path(args.out_json).resolve()}')
    print(out)


if __name__ == '__main__':
    main()
