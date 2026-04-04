from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import yaml

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def read_yaml(path: str | Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def write_yaml(path: str | Path, data: dict) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def resolve_path(base_yaml: str | Path, value: str | Path) -> Path:
    base_yaml = Path(base_yaml).resolve()
    value = Path(value)
    if value.is_absolute():
        return value
    return (base_yaml.parent / value).resolve()


def dataset_info(base_yaml: str | Path) -> Tuple[dict, Path, Dict[str, Path]]:
    cfg = read_yaml(base_yaml)
    root = resolve_path(base_yaml, cfg.get('path', '.')) if cfg.get('path') else Path(base_yaml).resolve().parent
    splits = {}
    for split in ('train', 'val', 'test'):
        if cfg.get(split):
            splits[split] = resolve_path(base_yaml, cfg[split])
    return cfg, root, splits


def list_images(image_dir: str | Path) -> List[Path]:
    image_dir = Path(image_dir)
    return sorted([p for p in image_dir.rglob('*') if p.suffix.lower() in IMG_EXTS])


def image_to_label_path(image_path: str | Path) -> Path:
    image_path = Path(image_path)
    parts = list(image_path.parts)
    if 'images' in parts:
        idx = parts.index('images')
        parts[idx] = 'labels'
        label_path = Path(*parts).with_suffix('.txt')
    else:
        label_path = image_path.with_suffix('.txt')
    return label_path


def read_yolo_boxes(label_path: str | Path) -> List[Tuple[int, float, float, float, float]]:
    label_path = Path(label_path)
    if not label_path.exists():
        return []
    rows = []
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:5])
            rows.append((cls, x, y, w, h))
    return rows


def yolo_box_to_xyxy(box: Sequence[float], width: int, height: int) -> Tuple[int, int, int, int]:
    x, y, w, h = box
    x1 = int(round((x - w / 2) * width))
    y1 = int(round((y - h / 2) * height))
    x2 = int(round((x + w / 2) * width))
    y2 = int(round((y + h / 2) * height))
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return x1, y1, x2, y2


def xyxy_to_yolo(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> Tuple[float, float, float, float]:
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    xc = x1 + bw / 2
    yc = y1 + bh / 2
    return xc / width, yc / height, bw / width, bh / height


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: str | Path, data: dict) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_rgb_mean_std(image_paths: Sequence[Path]) -> Tuple[List[float], List[float]]:
    n = 0
    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sq_sum = np.zeros(3, dtype=np.float64)
    for p in image_paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        flat = img.reshape(-1, 3)
        channel_sum += flat.sum(axis=0)
        channel_sq_sum += (flat ** 2).sum(axis=0)
        n += flat.shape[0]
    mean = channel_sum / max(n, 1)
    std = np.sqrt(channel_sq_sum / max(n, 1) - mean ** 2)
    return mean.tolist(), std.tolist()


def apply_global_norm_uint8(rgb: np.ndarray, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    img = rgb.astype(np.float32) / 255.0
    mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    std = np.clip(std, 1e-6, None)
    z = (img - mean) / std
    z = np.clip(z, -3.0, 3.0)
    out = ((z + 3.0) / 6.0 * 255.0).clip(0, 255).astype(np.uint8)
    return out


def bbox_union_xyxy(boxes: Sequence[Tuple[int, int, int, int]], width: int, height: int, expand_ratio: float = 0.08) -> Tuple[int, int, int, int]:
    if not boxes:
        return 0, 0, width, height
    xs1, ys1, xs2, ys2 = zip(*boxes)
    x1, y1, x2, y2 = min(xs1), min(ys1), max(xs2), max(ys2)
    pad_x = int(round((x2 - x1) * expand_ratio))
    pad_y = int(round((y2 - y1) * expand_ratio))
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(width, x2 + pad_x)
    y2 = min(height, y2 + pad_y)
    return x1, y1, x2, y2


def connected_bbox(mask: np.ndarray, expand_ratio: float = 0.08) -> Tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    h, w = mask.shape[:2]
    return bbox_union_xyxy([(x1, y1, x2, y2)], w, h, expand_ratio=expand_ratio)
