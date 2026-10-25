"""Run a documented YOLOv8 model on images and save predicted boxes.

Detection labels represent visible, dataset-defined defects only. Predictions
are not evidence of food safety or actual vacuum-seal integrity.
"""
import argparse
from pathlib import Path


def main() -> None:
    """Load student-trained weights and annotate provided image(s)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('runs_predictions'))
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--conf', type=float, default=.25)
    args = parser.parse_args()
    if not args.weights.is_file() or not args.source.exists():
        parser.error('Weight file and input image/directory must exist')
    if not 0 <= args.conf <= 1:
        parser.error('Confidence threshold must be from 0 to 1')
    from ultralytics import YOLO
    model = YOLO(str(args.weights))
    model.predict(source=str(args.source), imgsz=args.imgsz, conf=args.conf,
                  save=True, project=str(args.output_dir), name='prediction', exist_ok=True)
    print('Prediction images saved under:', args.output_dir / 'prediction')


if __name__ == '__main__':
    main()
