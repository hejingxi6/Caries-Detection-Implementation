from pathlib import Path

root = Path(r"D:\projects\yolo_fourth_round_opt\stage2_lesion3_v2")

for split in ["train", "val", "test"]:
    label_dir = root / split / "labels"
    files = list(label_dir.glob("*.txt"))
    total = len(files)
    empty = 0
    nonempty = 0
    box_counts = []

    for f in files:
        text = f.read_text(encoding="utf-8").strip()
        if text == "":
            empty += 1
        else:
            lines = [line for line in text.splitlines() if line.strip()]
            nonempty += 1
            box_counts.append(len(lines))

    empty_ratio = empty / total if total else 0
    avg_boxes = sum(box_counts) / len(box_counts) if box_counts else 0

    print(f"[{split}] total={total}, empty={empty}, empty_ratio={empty_ratio:.3f}, "
          f"nonempty={nonempty}, avg_boxes_nonempty={avg_boxes:.2f}")