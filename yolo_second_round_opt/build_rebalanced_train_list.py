import argparse
from collections import Counter
from pathlib import Path
import yaml


IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]


def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(base_dir: Path, raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def find_image_for_label(label_path: Path, images_dir: Path) -> Path | None:
    stem = label_path.stem
    for ext in IMG_EXTS:
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def parse_label_classes(label_path: Path) -> set[int]:
    classes = set()
    if not label_path.exists():
        return classes
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                classes.add(int(float(parts[0])))
            except ValueError:
                continue
    return classes


def main():
    parser = argparse.ArgumentParser(
        description="Build an oversampled train list for lesion-heavy YOLO training."
    )
    parser.add_argument("--data", required=True, help="Path to original data.yaml")
    parser.add_argument("--out_dir", default="rebalanced_data", help="Output folder")
    parser.add_argument(
        "--repeat_map",
        default="0:2,1:2,2:4",
        help="Repeat factor by class id, e.g. 0:2,1:2,2:4",
    )
    parser.add_argument(
        "--background_repeat",
        type=int,
        default=1,
        help="Repeat factor for images with no target classes from repeat_map",
    )
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    data = load_yaml(data_path)
    base_dir = data_path.parent

    train_raw = data.get("train")
    val_raw = data.get("val")
    test_raw = data.get("test")

    if train_raw is None or val_raw is None:
        raise ValueError("data.yaml must contain at least train and val.")

    train_path = resolve_path(base_dir, train_raw)
    val_path = resolve_path(base_dir, val_raw)
    test_path = resolve_path(base_dir, test_raw) if test_raw else None

    if not train_path.is_dir():
        raise ValueError(f"Expected train to be an image directory. Got: {train_path}")

    labels_dir = train_path.parent / "labels"
    if not labels_dir.exists():
        raise FileNotFoundError(f"Could not find labels directory: {labels_dir}")

    repeat_map = {}
    for item in args.repeat_map.split(","):
        item = item.strip()
        if not item:
            continue
        cls_id, repeat = item.split(":")
        repeat_map[int(cls_id)] = int(repeat)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_list_path = out_dir / "train_rebalanced.txt"
    stats_path = out_dir / "rebalanced_stats.txt"
    new_yaml_path = out_dir / "data_rebalanced.yaml"

    image_entries = []
    repeat_counter = Counter()
    class_image_counter = Counter()

    label_files = sorted(labels_dir.glob("*.txt"))
    if not label_files:
        raise FileNotFoundError(f"No label txt files found in {labels_dir}")

    for label_path in label_files:
        image_path = find_image_for_label(label_path, train_path)
        if image_path is None:
            continue

        classes = parse_label_classes(label_path)
        repeat = args.background_repeat
        hit = False
        for cls_id in classes:
            class_image_counter[cls_id] += 1
            if cls_id in repeat_map:
                repeat = max(repeat, repeat_map[cls_id])
                hit = True

        if not hit:
            repeat_counter["background_or_tooth_only"] += 1

        repeat_counter[f"repeat_{repeat}"] += 1
        for _ in range(repeat):
            image_entries.append(str(image_path))

    with open(train_list_path, "w", encoding="utf-8") as f:
        for item in image_entries:
            f.write(item + "\n")

    new_data = dict(data)
    new_data["train"] = str(train_list_path)
    new_data["val"] = str(val_path)
    if test_path:
        new_data["test"] = str(test_path)

    with open(new_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(new_data, f, allow_unicode=True, sort_keys=False)

    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("=== Rebalanced Train List Stats ===\n")
        f.write(f"Original train dir: {train_path}\n")
        f.write(f"Labels dir: {labels_dir}\n")
        f.write(f"Output train list: {train_list_path}\n")
        f.write(f"Repeat map: {repeat_map}\n")
        f.write(f"Background repeat: {args.background_repeat}\n")
        f.write(f"Total original labeled images: {len(label_files)}\n")
        f.write(f"Total train entries after repeat: {len(image_entries)}\n")
        f.write("\nClass image counts (images containing class):\n")
        for k, v in sorted(class_image_counter.items()):
            f.write(f"  class {k}: {v}\n")
        f.write("\nRepeat distribution:\n")
        for k, v in repeat_counter.items():
            f.write(f"  {k}: {v}\n")

    print("Done.")
    print(f"New data yaml: {new_yaml_path}")
    print(f"Train list: {train_list_path}")
    print(f"Stats: {stats_path}")


if __name__ == "__main__":
    main()
