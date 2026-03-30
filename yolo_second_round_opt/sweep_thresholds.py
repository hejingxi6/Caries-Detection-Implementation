import argparse
import csv
import json
from pathlib import Path

import yaml
from ultralytics import YOLO


def get_class_names(data_yaml: str):
    with open(data_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names", [])
    if isinstance(names, dict):
        return [names[k] for k in sorted(names.keys())]
    return names


def scalar(v):
    if hasattr(v, "item"):
        return float(v.item())
    return float(v)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep confidence/IoU for YOLOv8.")
    parser.add_argument("--weights", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--project", type=str, default="runs/first_round")
    parser.add_argument("--name", type=str, default="threshold_sweep")
    parser.add_argument("--confs", type=str, default="0.001,0.01,0.05,0.10,0.15,0.20,0.25")
    parser.add_argument("--ious", type=str, default="0.50,0.60,0.70")
    args = parser.parse_args()

    confs = [float(x.strip()) for x in args.confs.split(",")]
    ious = [float(x.strip()) for x in args.ious.split(",")]

    out_dir = Path(args.project) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    class_names = get_class_names(args.data)

    rows = []
    best_row = None
    best_score = -1.0

    for conf in confs:
        for iou in ious:
            metrics = model.val(
                data=args.data,
                imgsz=args.imgsz,
                batch=args.batch,
                device=args.device,
                split=args.split,
                conf=conf,
                iou=iou,
                plots=False,
                save_json=False,
                project=args.project,
                name=f"{args.name}_{args.split}_c{str(conf).replace('.', '')}_i{str(iou).replace('.', '')}",
                exist_ok=True,
                verbose=False,
            )

            row = {
                "split": args.split,
                "conf": conf,
                "iou": iou,
                "precision": scalar(metrics.box.mp),
                "recall": scalar(metrics.box.mr),
                "map50": scalar(metrics.box.map50),
                "map50_95": scalar(metrics.box.map),
                "fitness": scalar(metrics.fitness),
            }

            ap50_by_class = []
            ap_by_class = []
            if hasattr(metrics.box, "ap50") and metrics.box.ap50 is not None:
                ap50_by_class = [float(x) for x in metrics.box.ap50]
            if hasattr(metrics.box, "ap") and metrics.box.ap is not None:
                ap_by_class = [float(x) for x in metrics.box.ap]

            for idx, name in enumerate(class_names):
                row[f"ap50_{name}"] = ap50_by_class[idx] if idx < len(ap50_by_class) else None
                row[f"ap50_95_{name}"] = ap_by_class[idx] if idx < len(ap_by_class) else None

            rows.append(row)

            if row["fitness"] > best_score:
                best_score = row["fitness"]
                best_row = row

            print(
                f"split={args.split} conf={conf:.3f} iou={iou:.2f} "
                f"P={row['precision']:.4f} R={row['recall']:.4f} "
                f"mAP50={row['map50']:.4f} mAP50-95={row['map50_95']:.4f}"
            )

    csv_path = out_dir / f"sweep_{args.split}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(out_dir / f"best_{args.split}.json", "w", encoding="utf-8") as f:
        json.dump(best_row, f, indent=2)

    print("\nSweep complete.")
    print(f"CSV saved to: {csv_path}")
    print(f"Best row: {best_row}")


if __name__ == "__main__":
    main()
