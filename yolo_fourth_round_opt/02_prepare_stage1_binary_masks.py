from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

from common import dataset_info, ensure_dir, image_to_label_path, list_images, read_yolo_boxes, yolo_box_to_xyxy


def copy_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description='Build stage-1 binary segmentation dataset from YOLO box labels.')
    parser.add_argument('--data', required=True, help='Base YOLO dataset yaml')
    parser.add_argument('--out_root', required=True, help='Output root for binary seg dataset')
    parser.add_argument('--foreground_classes', default='all', help='Comma-separated class ids or all')
    args = parser.parse_args()

    _, _, splits = dataset_info(args.data)
    out_root = Path(args.out_root)
    use_all = args.foreground_classes.lower() == 'all'
    fg_classes = None if use_all else {int(x) for x in args.foreground_classes.split(',') if x.strip()}

    for split, image_dir in splits.items():
        out_img_dir = ensure_dir(out_root / split / 'images')
        out_mask_dir = ensure_dir(out_root / split / 'masks')
        images = list_images(image_dir)
        print(f'[{split}] building binary masks for {len(images)} images...')
        for img_path in images:
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            label_path = image_to_label_path(img_path)
            boxes = read_yolo_boxes(label_path)
            for cls, x, y, bw, bh in boxes:
                if fg_classes is not None and cls not in fg_classes:
                    continue
                x1, y1, x2, y2 = yolo_box_to_xyxy((x, y, bw, bh), w, h)
                mask[y1:y2, x1:x2] = 255
            copy_image(img_path, out_img_dir / img_path.name)
            cv2.imwrite(str(out_mask_dir / f'{img_path.stem}.png'), mask)
    print(f'Done. Stage-1 dataset saved to {out_root.resolve()}')


if __name__ == '__main__':
    main()
