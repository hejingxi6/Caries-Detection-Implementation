import argparse
import json
from pathlib import Path

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction, predict


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def gather_images(path_str: str):
    path = Path(path_str)
    if path.is_file():
        return [path]
    return [p for p in sorted(path.rglob("*")) if p.suffix.lower() in IMG_EXTS]


def main() -> None:
    parser = argparse.ArgumentParser(description="SAHI sliced inference for YOLOv8.")
    parser.add_argument("--weights", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--source", type=str, required=True, help="Image file or folder")
    parser.add_argument("--save_dir", type=str, default="runs/sahi_predict")
    parser.add_argument("--confidence", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--slice_height", type=int, default=512)
    parser.add_argument("--slice_width", type=int, default=512)
    parser.add_argument("--overlap_height_ratio", type=float, default=0.20)
    parser.add_argument("--overlap_width_ratio", type=float, default=0.20)
    parser.add_argument("--full_image_fallback", action="store_true", help="Also run plain inference for comparison")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=args.weights,
        confidence_threshold=args.confidence,
        device=args.device,
    )

    image_paths = gather_images(args.source)
    if not image_paths:
        raise FileNotFoundError(f"No images found under: {args.source}")

    summary = []
    for img_path in image_paths:
        result = get_sliced_prediction(
            str(img_path),
            detection_model,
            slice_height=args.slice_height,
            slice_width=args.slice_width,
            overlap_height_ratio=args.overlap_height_ratio,
            overlap_width_ratio=args.overlap_width_ratio,
            verbose=0,
        )

        stem_dir = save_dir / img_path.stem
        stem_dir.mkdir(parents=True, exist_ok=True)

        result.export_visuals(
            export_dir=str(stem_dir),
            file_name="sahi_prediction",
            rect_th=2,
            text_size=0.6,
            hide_labels=False,
            hide_conf=False,
        )
        result.to_coco_predictions(image_id=img_path.name)
        result_json_path = stem_dir / "sahi_prediction.json"
        with open(result_json_path, "w", encoding="utf-8") as f:
            json.dump(result.to_coco_predictions(image_id=img_path.name), f, indent=2)

        item = {
            "image": str(img_path),
            "num_predictions": len(result.object_prediction_list),
            "output_dir": str(stem_dir),
        }

        if args.full_image_fallback:
            plain_result = predict(
                detection_model=detection_model,
                source=str(img_path),
                no_standard_prediction=False,
                no_sliced_prediction=True,
                export_visual=True,
                export_pickle=False,
                export_crop=False,
                project=str(stem_dir),
                name="plain_prediction",
                verbose=0,
            )
            item["plain_prediction_dir"] = str(stem_dir / "plain_prediction")

        summary.append(item)
        print(f"Processed: {img_path} | predictions={item['num_predictions']}")

    with open(save_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSAHI inference complete.")
    print(f"Saved to: {save_dir}")


if __name__ == "__main__":
    main()
