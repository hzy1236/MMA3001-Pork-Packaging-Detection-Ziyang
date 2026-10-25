"""Evaluate a trained detector on validation or held-out test images.

Test data must not be used to tune hyperparameters. The resulting P/R/AP
metrics describe object detection, *not* image-level classification accuracy.
"""
import argparse
import json
from pathlib import Path


def extract_metrics(box_metrics: object) -> dict[str, float]:
    """Convert Ultralytics detection box metrics to serialisable floats.

    Args:
        box_metrics: ``model.val(...).box`` object with ``mp``, ``mr``, ``map50``, ``map``.

    Returns:
        Mean precision, recall, mAP at IoU .5, mAP across IoU .5–.95.
    """
    return {"precision": float(box_metrics.mp), "recall": float(box_metrics.mr),
            "mAP50": float(box_metrics.map50), "mAP50_95": float(box_metrics.map)}


def main() -> None:
    """Load saved weights, compute requested split's metrics and write JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--imgsz", type=int, choices=(416, 640), default=640)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.weights.is_file() or not args.data.is_file():
        parser.error("Both model checkpoint and dataset config must exist")
    from ultralytics import YOLO
    metrics = YOLO(str(args.weights)).val(data=str(args.data), split=args.split, imgsz=args.imgsz, device=0, plots=True)
    report = {"split": args.split, "imgsz": args.imgsz, "metrics": extract_metrics(metrics.box)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
