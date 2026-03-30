import argparse
import json
from pathlib import Path

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def gather_images(path_str: str):
    path = Path(path_str)
    if path.is_file():
        return [path]
    return [p for p in sorted(path.rglob("*")) if p.suffix.lower() in IMG_EXTS]


def main():
    parser = argparse.ArgumentParser(description="Optional SAHI eval for round 3.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--save_dir", default="runs/sahi_round3")
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--slice_height", type=int, default=512)
    parser.add_argument("--slice_width", type=int, default=512)
    parser.add_argument("--overlap_height_ratio", type=float, default=0.20)
    parser.add_argument("--overlap_width_ratio", type=float, default=0.20)
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=args.weights,
        confidence_threshold=args.confidence,
        device=args.device,
    )

    summary = []
    for img_path in gather_images(args.source):
        result = get_sliced_prediction(
            str(img_path),
            detection_model,
            slice_height=args.slice_height,
            slice_width=args.slice_width,
            overlap_height_ratio=args.overlap_height_ratio,
            overlap_width_ratio=args.overlap_width_ratio,
            verbose=0,
        )
        out_dir = save_dir / img_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        result.export_visuals(
            export_dir=str(out_dir),
            file_name="sahi_prediction",
            rect_th=2,
            text_size=0.6,
            hide_labels=False,
            hide_conf=False,
        )
        coco_preds = result.to_coco_predictions(image_id=img_path.name)
        with open(out_dir / "sahi_prediction.json", "w", encoding="utf-8") as f:
            json.dump(coco_preds, f, indent=2)
        summary.append(
            {
                "image": str(img_path),
                "num_predictions": len(result.object_prediction_list),
                "output_dir": str(out_dir),
            }
        )

    with open(save_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Done. Summary saved to {save_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
