from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2

from common import apply_global_norm_uint8, dataset_info, ensure_dir, image_to_label_path, list_images, load_json, read_yolo_boxes, write_yaml, xyxy_to_yolo, yolo_box_to_xyxy


def save_labels(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for cls, x, y, w, h in rows:
            f.write(f'{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n')


def main():
    parser = argparse.ArgumentParser(description='Build stage-2 lesion-only dataset from stage-1 ROI crops.')
    parser.add_argument('--data', required=True)
    parser.add_argument('--roi_json', required=True)
    parser.add_argument('--out_root', required=True)
    parser.add_argument('--lesion_classes', default='0,1,2')
    parser.add_argument('--tooth_class', type=int, default=3)
    parser.add_argument('--expand_ratio', type=float, default=0.10, help='Extra pad on top of stage1 roi before crop')
    parser.add_argument('--norm_json', default='')
    parser.add_argument('--apply_global_norm', action='store_true')
    args = parser.parse_args()

    lesion_classes = [int(x) for x in args.lesion_classes.split(',') if x.strip()]
    lesion_map = {old: new for new, old in enumerate(lesion_classes)}
    roi_db = load_json(args.roi_json)
    norm_data = load_json(args.norm_json) if args.norm_json else None

    _, _, splits = dataset_info(args.data)
    out_root = Path(args.out_root)

    for split, image_dir in splits.items():
        out_img_dir = ensure_dir(out_root / split / 'images')
        out_lab_dir = ensure_dir(out_root / split / 'labels')
        images = list_images(image_dir)
        print(f'[{split}] building stage-2 crops for {len(images)} images...')
        for img_path in images:
            abs_key = str(img_path.resolve())
            if abs_key not in roi_db:
                continue
            bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            x1, y1, x2, y2 = roi_db[abs_key]['bbox_xyxy']
            pad_x = int(round((x2 - x1) * args.expand_ratio))
            pad_y = int(round((y2 - y1) * args.expand_ratio))
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
            crop = rgb[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            if args.apply_global_norm and norm_data is not None:
                crop = apply_global_norm_uint8(crop, norm_data['train_mean_rgb'], norm_data['train_std_rgb'])
            crop_h, crop_w = crop.shape[:2]

            label_rows = []
            for cls, xc, yc, bw, bh in read_yolo_boxes(image_to_label_path(img_path)):
                if cls not in lesion_map:
                    continue
                bx1, by1, bx2, by2 = yolo_box_to_xyxy((xc, yc, bw, bh), w, h)
                nx1 = max(0, bx1 - x1)
                ny1 = max(0, by1 - y1)
                nx2 = min(crop_w, bx2 - x1)
                ny2 = min(crop_h, by2 - y1)
                if nx2 <= nx1 or ny2 <= ny1:
                    continue
                yolo = xyxy_to_yolo(nx1, ny1, nx2, ny2, crop_w, crop_h)
                label_rows.append((lesion_map[cls], *yolo))

            out_img_path = out_img_dir / img_path.name
            cv2.imwrite(str(out_img_path), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            save_labels(out_lab_dir / f'{img_path.stem}.txt', label_rows)

    yaml_data = {
        'path': str(out_root.resolve()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images' if (out_root / 'test' / 'images').exists() else None,
        'names': ['caries', 'cavity', 'crack'],
    }
    if yaml_data['test'] is None:
        yaml_data.pop('test')
    write_yaml(out_root / 'stage2_lesion3.yaml', yaml_data)
    print(f'Saved stage-2 lesion dataset to {out_root.resolve()}')
    print(f'New yaml: {out_root / "stage2_lesion3.yaml"}')


if __name__ == '__main__':
    main()
