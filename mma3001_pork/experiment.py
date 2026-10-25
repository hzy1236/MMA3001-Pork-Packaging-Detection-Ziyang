"""Reproduce YOLOv8n training used for the two-resolution comparison.

The student's original executed evidence is their Colab notebook/logs. This
module packages the same documented configuration for reproducibility.
"""
import argparse
from pathlib import Path

CONFIGS = {"baseline": {"imgsz": 416, "name": "yolov8n_baseline"}, "highres": {"imgsz": 640, "name": "yolov8n_640"}}


def training_arguments(variant: str, data: Path, runs: Path) -> dict:
    """Return documented training options, changing only input image size.

    Args:
        variant: One of ``baseline`` and ``highres``.
        data: Absolute-path YOLO dataset YAML.
        runs: Output directory for weights, losses and plots.

    Returns:
        Dictionary accepted by ``ultralytics.YOLO.train``.
    """
    if variant not in CONFIGS:
        raise ValueError(f"Unsupported variant: {variant}")
    return {
        "data": str(data), "epochs": 12, "imgsz": CONFIGS[variant]["imgsz"],
        "batch": 8, "device": 0, "seed": 42, "plots": True,
        "project": str(runs), "name": CONFIGS[variant]["name"],
    }


def main() -> None:
    """Fine-tune a pretrained YOLOv8n model on the selected dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=CONFIGS.keys(), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error("YOLO data config missing")
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    model.train(**training_arguments(args.variant, args.data, args.runs))


if __name__ == "__main__":
    main()
