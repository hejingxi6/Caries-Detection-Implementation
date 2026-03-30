import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from ultralytics import YOLO


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def read_num_classes(data_yaml: str) -> int:
    with open(data_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names", [])
    if isinstance(names, dict):
        return len(names)
    return len(names)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="First-round YOLOv8 optimization for dental lesion detection."
    )
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml")
    parser.add_argument("--weights", type=str, default="yolov8m.pt", help="Initial weights")
    parser.add_argument("--project", type=str, default="runs/first_round", help="Output root")
    parser.add_argument("--name", type=str, default="imgsz960_stage1", help="Run name")
    parser.add_argument("--imgsz", type=int, default=960, help="Training image size")
    parser.add_argument("--epochs", type=int, default=100, help="Epoch count")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default="0", help="CUDA device, e.g. 0 or cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--optimizer", type=str, default="AdamW", help="SGD/Adam/AdamW/auto")
    parser.add_argument("--lr0", type=float, default=0.0005, help="Initial learning rate")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final LR factor")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="Weight decay")
    parser.add_argument("--warmup_epochs", type=float, default=3.0, help="Warmup epochs")
    parser.add_argument("--box", type=float, default=7.5, help="Box loss gain")
    parser.add_argument("--cls", type=float, default=1.0, help="Class loss gain")
    parser.add_argument("--dfl", type=float, default=1.5, help="DFL loss gain")
    parser.add_argument("--close_mosaic", type=int, default=10, help="Disable mosaic in final epochs")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained backbone")
    parser.add_argument("--freeze", type=int, default=0, help="Freeze first N layers")
    parser.add_argument("--single_cls", action="store_true", help="Train as single-class detector")
    args = parser.parse_args()

    set_seed(args.seed)

    out_dir = Path(args.project) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    nc = read_num_classes(args.data)
    # Heuristic: small-object / imbalance-friendly augmentation.
    # This does not replace label cleanup or explicit class-balanced sampling,
    # but it is low-risk and fast for a first engineering round.
    hyp = {
        "lr0": args.lr0,
        "lrf": args.lrf,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "box": args.box,
        "cls": args.cls,
        "dfl": args.dfl,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 3.0,
        "translate": 0.08,
        "scale": 0.20,
        "shear": 1.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.10,
        "copy_paste": 0.20,
        "erasing": 0.15,
    }

    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    with open(out_dir / "augmentation_config.json", "w", encoding="utf-8") as f:
        json.dump(hyp, f, indent=2)

    model = YOLO(args.weights)

    results = model.train(
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
        freeze=args.freeze,
        single_cls=args.single_cls,
        amp=True,
        cache=False,
        cos_lr=True,
        multi_scale=False,
        overlap_mask=False,
        rect=False,
        val=True,
        save=True,
        save_period=-1,
        plots=True,
        verbose=True,
        **hyp,
    )

    best_weight = Path(args.project) / args.name / "weights" / "best.pt"
    val_metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=0.001,
        iou=0.7,
        split="val",
        save_json=False,
        plots=True,
        project=args.project,
        name=f"{args.name}_val_default",
        exist_ok=True,
    )

    summary = {
        "best_weight_exists": best_weight.exists(),
        "best_weight": str(best_weight),
        "nc": nc,
        "note": "Use sweep_thresholds.py next. The default validation threshold is not your final operating point.",
    }
    with open(out_dir / "post_train_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nTraining finished.")
    print(f"Best weight: {best_weight}")
    print("Now run threshold sweep on val/test before deciding final operating point.")


if __name__ == "__main__":
    main()
