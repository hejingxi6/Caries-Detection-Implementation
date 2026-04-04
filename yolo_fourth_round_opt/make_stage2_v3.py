from pathlib import Path
import shutil
import random

random.seed(42)

SRC_ROOT = Path(r"D:\projects\yolo_fourth_round_opt\stage2_lesion3_v2")
DST_ROOT = Path(r"D:\projects\yolo_fourth_round_opt\stage2_lesion3_v3")

# 保留策略：全部非空 + 限量空样本
EMPTY_KEEP = {
    "train": 200,
    "val": 20,
    "test": 25,
}

CLASS_NAMES = ["Caries", "Cavity", "Crack"]

def read_label_file(p: Path) -> str:
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8").strip()

def copy_pair(img_path: Path, src_label: Path, dst_img_dir: Path, dst_label_dir: Path):
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_label_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(img_path, dst_img_dir / img_path.name)
    if src_label.exists():
        shutil.copy2(src_label, dst_label_dir / src_label.name)
    else:
        # 如果没有 label 文件，就创建空文件
        (dst_label_dir / (img_path.stem + ".txt")).write_text("", encoding="utf-8")

def main():
    if DST_ROOT.exists():
        shutil.rmtree(DST_ROOT)
    DST_ROOT.mkdir(parents=True, exist_ok=True)

    summary = {}

    for split in ["train", "val", "test"]:
        src_img_dir = SRC_ROOT / split / "images"
        src_label_dir = SRC_ROOT / split / "labels"

        dst_img_dir = DST_ROOT / split / "images"
        dst_label_dir = DST_ROOT / split / "labels"

        images = sorted([p for p in src_img_dir.glob("*") if p.is_file()])

        positives = []
        empties = []

        for img in images:
            # 跳过 checkpoint 重复文件
            if "-checkpoint" in img.name:
                continue

            label = src_label_dir / f"{img.stem}.txt"
            txt = read_label_file(label)

            if txt == "":
                empties.append((img, label))
            else:
                positives.append((img, label))

        keep_empty_n = min(EMPTY_KEEP[split], len(empties))
        kept_empties = random.sample(empties, keep_empty_n)

        for img, label in positives:
            copy_pair(img, label, dst_img_dir, dst_label_dir)

        for img, label in kept_empties:
            copy_pair(img, label, dst_img_dir, dst_label_dir)

        total = len(positives) + len(kept_empties)
        empty_ratio = len(kept_empties) / total if total else 0.0

        summary[split] = {
            "positives": len(positives),
            "empties_kept": len(kept_empties),
            "total": total,
            "empty_ratio": empty_ratio,
        }

    yaml_text = f"""train: {DST_ROOT / 'train' / 'images'}
val: {DST_ROOT / 'val' / 'images'}
test: {DST_ROOT / 'test' / 'images'}
nc: 3
names:
  - {CLASS_NAMES[0]}
  - {CLASS_NAMES[1]}
  - {CLASS_NAMES[2]}
"""
    (DST_ROOT / "stage2_lesion3_v3.yaml").write_text(yaml_text, encoding="utf-8")

    print("=== stage2_lesion3_v3 built ===")
    for split, info in summary.items():
        print(
            f"[{split}] positives={info['positives']}, "
            f"empties_kept={info['empties_kept']}, "
            f"total={info['total']}, "
            f"empty_ratio={info['empty_ratio']:.3f}"
        )
    print(f"YAML saved to: {DST_ROOT / 'stage2_lesion3_v3.yaml'}")

if __name__ == "__main__":
    main()