from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from common import connected_bbox, dataset_info, ensure_dir, list_images, save_json
from tiny_unet import TinyUNet


def preprocess(img_rgb: np.ndarray, size: int, mean, std):
    x = cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    std = np.clip(std, 1e-6, None).astype(np.float32)
    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))[None, ...].astype(np.float32)
    return torch.from_numpy(x).float()


def main():
    parser = argparse.ArgumentParser(description='Predict stage-1 binary masks and ROIs for all splits.')
    parser.add_argument('--data', required=True)
    parser.add_argument('--weights', required=True, help='best_stage1.pth')
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--expand_ratio', type=float, default=0.08)
    args = parser.parse_args()

    ckpt = torch.load(args.weights, map_location='cpu')
    train_args = ckpt['args']
    image_size = int(train_args['image_size'])
    mean = ckpt.get('mean', [0.5, 0.5, 0.5])
    std = ckpt.get('std', [0.5, 0.5, 0.5])

    model = TinyUNet(base=train_args.get('base_channels', 32))
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device=device, dtype=torch.float32)

    _, _, splits = dataset_info(args.data)
    out_dir = Path(args.out_dir)
    roi_db = {}

    for split, image_dir in splits.items():
        mask_dir = ensure_dir(out_dir / split / 'masks')
        images = list_images(image_dir)
        print(f'[{split}] predicting stage-1 masks for {len(images)} images...')
        for img_path in tqdm(images, leave=False):
            bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            x = preprocess(rgb, image_size, mean, std).to(device=device, dtype=torch.float32)
            with torch.no_grad():
                logits = model(x)
                prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
            mask_small = (prob > args.threshold).astype(np.uint8) * 255
            mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(mask_dir / f'{img_path.stem}.png'), mask)
            bbox = connected_bbox(mask, expand_ratio=args.expand_ratio)
            if bbox is None:
                bbox = [0, 0, w, h]
            roi_db[str(img_path.resolve())] = {'split': split, 'bbox_xyxy': list(map(int, bbox))}

    save_json(out_dir / 'stage1_rois.json', roi_db)
    print(f'Saved stage-1 masks + ROI json to {out_dir.resolve()}')


if __name__ == '__main__':
    main()
