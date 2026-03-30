import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image


IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]


def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(base_dir: Path, raw: str | None) -> Path | None:
    if raw is None:
        return None
    p = Path(raw)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def parse_map(text: str) -> dict[int, int]:
    result = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        k, v = item.split(":")
        result[int(k)] = int(v)
    return result


def parse_int_set(text: str) -> set[int]:
    return {int(x.strip()) for x in text.split(",") if x.strip()}


def find_image_for_label(label_path: Path, images_dir: Path) -> Path | None:
    stem = label_path.stem
    for ext in IMG_EXTS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def read_labels(label_path: Path):
    rows = []
    if not label_path.exists():
        return rows
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            try:
                cls_id = int(float(parts[0]))
                xc, yc, w, h = map(float, parts[1:])
                rows.append((cls_id, xc, yc, w, h))
            except ValueError:
                continue
    return rows


def yolo_to_xyxy(box, img_w, img_h):
    cls_id, xc, yc, w, h = box
    bw = w * img_w
    bh = h * img_h
    x1 = (xc * img_w) - bw / 2
    y1 = (yc * img_h) - bh / 2
    x2 = x1 + bw
    y2 = y1 + bh
    return cls_id, x1, y1, x2, y2


def clip_box(x1, y1, x2, y2, crop_x1, crop_y1, crop_x2, crop_y2):
    nx1 = max(x1, crop_x1)
    ny1 = max(y1, crop_y1)
    nx2 = min(x2, crop_x2)
    ny2 = min(y2, crop_y2)
    if nx2 <= nx1 or ny2 <= ny1:
        return None
    return nx1, ny1, nx2, ny2


def to_yolo_from_crop(cls_id, x1, y1, x2, y2, crop_x1, crop_y1, crop_w, crop_h):
    bx1 = x1 - crop_x1
    by1 = y1 - crop_y1
    bx2 = x2 - crop_x1
    by2 = y2 - crop_y1
    bw = bx2 - bx1
    bh = by2 - by1
    xc = bx1 + bw / 2
    yc = by1 + bh / 2
    return (
        cls_id,
        xc / crop_w,
        yc / crop_h,
        bw / crop_w,
        bh / crop_h,
    )


def make_square_crop(cx, cy, side, img_w, img_h):
    side = min(side, max(img_w, img_h))
    x1 = int(round(cx - side / 2))
    y1 = int(round(cy - side / 2))
    x2 = x1 + int(side)
    y2 = y1 + int(side)

    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > img_w:
        shift = x2 - img_w
        x1 -= shift
        x2 = img_w
    if y2 > img_h:
        shift = y2 - img_h
        y1 -= shift
        y2 = img_h

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w, x2)
    y2 = min(img_h, y2)

    # If image is rectangular, pad by shrinking to valid square-like region if needed.
    crop_w = x2 - x1
    crop_h = y2 - y1
    side2 = min(crop_w, crop_h)
    x2 = x1 + side2
    y2 = y1 + side2
    return x1, y1, x2, y2


def main():
    parser = argparse.ArgumentParser(
        description="Build a third-round lesion-aware mixed training set for YOLOv8."
    )
    parser.add_argument("--data", required=True, help="Path to original data.yaml")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument(
        "--repeat_map",
        default="0:3,1:2,2:6",
        help="Original-image repeat by class id, e.g. 0:3,1:2,2:6",
    )
    parser.add_argument(
        "--crop_repeat_map",
        default="0:2,1:2,2:4",
        help="Crop repeat by lesion class id",
    )
    parser.add_argument(
        "--lesion_classes",
        default="0,1,2",
        help="Comma-separated lesion class ids",
    )
    parser.add_argument("--background_repeat", type=int, default=1)
    parser.add_argument("--multi_lesion_bonus", type=int, default=1)
    parser.add_argument("--min_crop_size", type=int, default=384)
    parser.add_argument("--max_crop_size", type=int, default=960)
    parser.add_argument("--context_scale", type=float, default=7.0)
    parser.add_argument("--keep_area_frac", type=float, default=0.40)
    parser.add_argument("--min_box_px", type=float, default=4.0)
    parser.add_argument("--max_crops_per_image", type=int, default=6)
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    data = load_yaml(data_path)
    base_dir = data_path.parent

    train_path = resolve_path(base_dir, data.get("train"))
    val_path = resolve_path(base_dir, data.get("val"))
    test_path = resolve_path(base_dir, data.get("test"))

    if train_path is None or not train_path.is_dir():
        raise ValueError(f"Expected train to be an image directory. Got: {train_path}")

    labels_dir = train_path.parent / "labels"
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    repeat_map = parse_map(args.repeat_map)
    crop_repeat_map = parse_map(args.crop_repeat_map)
    lesion_classes = parse_int_set(args.lesion_classes)

    out_dir = Path(args.out_dir).resolve()
    crops_img_dir = out_dir / "round3_crops" / "images"
    crops_lbl_dir = out_dir / "round3_crops" / "labels"
    crops_img_dir.mkdir(parents=True, exist_ok=True)
    crops_lbl_dir.mkdir(parents=True, exist_ok=True)

    train_list = []
    stats = {
        "original_entries": 0,
        "crop_entries": 0,
        "total_train_entries": 0,
        "generated_crops": 0,
    }
    image_repeat_stats = Counter()
    lesion_crop_stats = Counter()
    class_presence = Counter()
    per_image_crop_count = defaultdict(int)

    label_files = sorted(labels_dir.glob("*.txt"))
    if not label_files:
        raise FileNotFoundError(f"No label files found in: {labels_dir}")

    for label_path in label_files:
        image_path = find_image_for_label(label_path, train_path)
        if image_path is None:
            continue

        labels = read_labels(label_path)
        if not labels:
            train_list.extend([str(image_path)] * args.background_repeat)
            image_repeat_stats[f"repeat_{args.background_repeat}"] += 1
            stats["original_entries"] += args.background_repeat
            continue

        lesion_rows = [row for row in labels if row[0] in lesion_classes]
        classes_in_image = {row[0] for row in labels}
        for cls_id in classes_in_image:
            class_presence[cls_id] += 1

        repeat = args.background_repeat
        for cls_id in {row[0] for row in lesion_rows}:
            repeat = max(repeat, repeat_map.get(cls_id, args.background_repeat))
        if len(lesion_rows) >= 2:
            repeat += args.multi_lesion_bonus

        train_list.extend([str(image_path)] * repeat)
        image_repeat_stats[f"repeat_{repeat}"] += 1
        stats["original_entries"] += repeat

        if not lesion_rows:
            continue

        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img_w, img_h = img.size

            # Prioritize smaller lesions first because small-lesion recall is the bottleneck.
            lesion_rows_sorted = sorted(
                lesion_rows,
                key=lambda r: (r[3] * r[4], r[0])
            )[: args.max_crops_per_image]

            pixel_boxes = [yolo_to_xyxy(box, img_w, img_h) for box in labels]

            for crop_idx, lesion in enumerate(lesion_rows_sorted):
                cls_id, xc, yc, bw, bh = lesion
                lesion_w = bw * img_w
                lesion_h = bh * img_h
                side = max(
                    args.min_crop_size,
                    int(math.ceil(max(lesion_w, lesion_h) * args.context_scale))
                )
                side = min(side, args.max_crop_size)

                crop_x1, crop_y1, crop_x2, crop_y2 = make_square_crop(
                    xc * img_w,
                    yc * img_h,
                    side,
                    img_w,
                    img_h,
                )
                crop_w = crop_x2 - crop_x1
                crop_h = crop_y2 - crop_y1
                if crop_w <= 1 or crop_h <= 1:
                    continue

                new_labels = []
                lesion_present_after_crop = False
                for px in pixel_boxes:
                    p_cls, x1, y1, x2, y2 = px
                    clipped = clip_box(x1, y1, x2, y2, crop_x1, crop_y1, crop_x2, crop_y2)
                    if clipped is None:
                        continue
                    nx1, ny1, nx2, ny2 = clipped

                    orig_area = max(1e-6, (x2 - x1) * (y2 - y1))
                    kept_area = max(0.0, (nx2 - nx1) * (ny2 - ny1))
                    if kept_area / orig_area < args.keep_area_frac:
                        continue
                    if (nx2 - nx1) < args.min_box_px or (ny2 - ny1) < args.min_box_px:
                        continue

                    yolo_box = to_yolo_from_crop(
                        p_cls, nx1, ny1, nx2, ny2, crop_x1, crop_y1, crop_w, crop_h
                    )
                    new_labels.append(yolo_box)
                    if p_cls == cls_id and p_cls in lesion_classes:
                        lesion_present_after_crop = True

                if not lesion_present_after_crop:
                    continue

                crop_img = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                crop_name = f"{image_path.stem}_crop{crop_idx}_cls{cls_id}.jpg"
                crop_img_path = crops_img_dir / crop_name
                crop_lbl_path = crops_lbl_dir / f"{Path(crop_name).stem}.txt"

                crop_img.save(crop_img_path, quality=95)

                with open(crop_lbl_path, "w", encoding="utf-8") as f:
                    for nl in new_labels:
                        f.write(
                            f"{nl[0]} {nl[1]:.6f} {nl[2]:.6f} {nl[3]:.6f} {nl[4]:.6f}\n"
                        )

                crop_repeat = crop_repeat_map.get(cls_id, 1)
                train_list.extend([str(crop_img_path)] * crop_repeat)
                stats["crop_entries"] += crop_repeat
                stats["generated_crops"] += 1
                lesion_crop_stats[f"class_{cls_id}_crop_repeat_{crop_repeat}"] += 1
                per_image_crop_count[image_path.name] += 1

    train_txt = out_dir / "train_round3.txt"
    with open(train_txt, "w", encoding="utf-8") as f:
        for p in train_list:
            f.write(p + "\n")

    new_data = dict(data)
    new_data["train"] = str(train_txt)
    new_data["val"] = str(val_path) if val_path else data.get("val")
    if test_path:
        new_data["test"] = str(test_path)

    new_yaml = out_dir / "data_round3.yaml"
    with open(new_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(new_data, f, allow_unicode=True, sort_keys=False)

    stats["total_train_entries"] = len(train_list)
    stats["repeat_map"] = repeat_map
    stats["crop_repeat_map"] = crop_repeat_map
    stats["lesion_classes"] = sorted(list(lesion_classes))
    stats["class_presence"] = dict(class_presence)
    stats["image_repeat_stats"] = dict(image_repeat_stats)
    stats["lesion_crop_stats"] = dict(lesion_crop_stats)
    stats["top_cropped_images"] = sorted(
        per_image_crop_count.items(), key=lambda kv: kv[1], reverse=True
    )[:20]

    with open(out_dir / "round3_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    with open(out_dir / "round3_stats.txt", "w", encoding="utf-8") as f:
        f.write("=== Round 3 Lesion Mix Stats ===\n")
        f.write(f"Original train dir: {train_path}\n")
        f.write(f"Original labels dir: {labels_dir}\n")
        f.write(f"Generated yaml: {new_yaml}\n")
        f.write(f"Generated train list: {train_txt}\n")
        f.write(f"Original repeated entries: {stats['original_entries']}\n")
        f.write(f"Crop repeated entries: {stats['crop_entries']}\n")
        f.write(f"Generated crops: {stats['generated_crops']}\n")
        f.write(f"Total train entries: {stats['total_train_entries']}\n")
        f.write(f"Repeat map: {repeat_map}\n")
        f.write(f"Crop repeat map: {crop_repeat_map}\n")
        f.write(f"Lesion classes: {sorted(list(lesion_classes))}\n")

    print("Done.")
    print(f"New YAML: {new_yaml}")
    print(f"Train list: {train_txt}")
    print(f"Stats: {out_dir / 'round3_stats.txt'}")


if __name__ == "__main__":
    main()
