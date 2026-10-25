# MMA3001 — Experimental evidence

These files were selected without modifying the image content from the original Ultralytics run archives:
`yolov8n_baseline.zip`, `yolov8n_640.zip`, and `final_test_640.zip`.

## Experiments

- Baseline: YOLOv8n, 416×416, 12 epochs, seed 42, batch size 8, T4 GPU. Validation 120 images.
- Resolution experiment: YOLOv8n, 640×640, otherwise the same key settings. Validation 120 images.
- Final test: the selected 640px `best.pt` evaluated on an 80-image held-out test split.

The original Roboflow dataset is **Pork Rasher Error (Packaging), version 4**, from Roboflow Universe; use its dataset page and CC BY 4.0 attribution when reproducing figures: https://universe.roboflow.com/hello-2aqe0/pork-rasher-error-packaging

## Key metrics

| Split / experiment | Precision | Recall | mAP50 | mAP50–95 |
| --- | ---: | ---: | ---: | ---: |
| 416px validation | 0.463 | 0.386 | 0.450 | 0.288 |
| 640px validation | 0.659 | 0.486 | 0.478 | 0.261 |
| 640px test | 0.5063 | 0.4136 | 0.3553 | 0.1969 |

The higher resolution improved validation precision, recall, and mAP50 but **not** mAP50–95, and required longer training (about 7.2 vs 10.0 minutes). These validation and test numbers are **not interchangeable** and must not be treated as independent repeated trials.

## File guide

- `figures/baseline_416/results.png` and `figures/improved_640/results.png`: training curves.
- `figures/*/confusion_matrix_normalized.png`: normalised confusion matrices from corresponding runs.
- `figures/*/BoxPR_curve.png`: precision–recall curves.
- `figures/*/val_batch0_pred.jpg`: example box predictions generated during validation/test.
- `metrics/*training_log.csv`: raw Ultralytics epoch data.
- `metrics/comparison.csv`: summary metrics transcribed from your run logs.
- `metrics/class_counts.csv`: annotation counts per data split from the user's Colab class-count output.
- `metrics/final_test_per_class.csv`: classwise test metrics transcribed from the final test output.

## Limitations

- Imbalanced data: 0 labelled `wrinkle` instances in validation **and** test, so this class cannot be evaluated meaningfully.
- `loose-meat` and `twisted-meat` have only 3 and 4 test instances; classwise estimates are uncertain.
- Test mAP50 (0.3553) was below validation mAP50 (0.478), suggesting limited generalisation on this split.
- Similar sequential frames may occur across splits; independence needs verification.
- Visible RGB defects are not proof of food safety or seal integrity.
- The pretrained YOLOv8 architecture and dataset are third-party; clearly attribute them in the project report and README.

This curated folder **excludes** the model `.pt` weight files, private Roboflow credentials, and original image dataset. The complete model archives were saved separately in Google Drive by the student.
