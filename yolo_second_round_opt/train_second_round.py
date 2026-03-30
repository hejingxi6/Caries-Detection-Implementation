import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    parser = argparse.ArgumentParser(
        description="Second-round YOLOv8 training focused on lesion classes."
    )
    parser.add_argument("--data", required=True, help="Path to rebalanced data yaml")
    parser.add_argument("--weights", default="yolov8m.pt")
    parser.add_argument("--project", default="runs/second_round")
    parser.add_argument("--name", default="rebalance_imgsz768")
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=0.0004)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--warmup_epochs", type=float, default=3.0)
    parser.add_argument("--box", type=float, default=7.5)
    parser.add_argument("--cls", type=float, default=1.8)
    parser.add_argument("--dfl", type=float, default=1.5)
    parser.add_argument("--close_mosaic", type=int, default=10)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.project) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    hyp = {
        "lr0": args.lr0,
        "lrf": args.lrf,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "box": args.box,
        "cls": args.cls,
        "dfl": args.dfl,
        "hsv_h": 0.015,
        "hsv_s": 0.6,
        "hsv_v": 0.35,
        "degrees": 2.0,
        "translate": 0.05,
        "scale": 0.15,
        "shear": 0.5,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 0.8,
        "mixup": 0.05,
        "copy_paste": 0.15,
        "erasing": 0.10,
    }

    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    with open(out_dir / "augmentation_config.json", "w", encoding="utf-8") as f:
        json.dump(hyp, f, indent=2)

    model = YOLO(args.weights)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        pretrained=args.pretrained,
        optimizer=args.optimizer,
        patience=args.patience,
        project=args.project,
        name=args.name,
        exist_ok=True,
        close_mosaic=args.close_mosaic,
        amp=True,
        cache=False,
        cos_lr=True,
        multi_scale=False,
        overlap_mask=False,
        rect=False,
        val=True,
        save=True,
        plots=True,
        verbose=True,
        **hyp,
    )

    best_weight = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"Training finished. Best weight: {best_weight}")


if __name__ == "__main__":
    main()
