# MMA3001 — RGB-Based Pork Packaging Defect Detection Using YOLOv8n

**Individual project | Numerical Methods and Machine Learning | Object detection**

## Project overview

This project investigates whether a pretrained YOLOv8n object detector can identify five labelled types of visible defects in pork-rasher packaging images. It also compares two input resolutions (416 × 416 and 640 × 640 pixels) to examine the trade-off between detection performance and training time. The goal is an **inspection-assistance proof of concept**, not a production-ready accept/reject system. Image predictions cannot establish food safety or physical seal integrity.

## Dataset

- **Source:** Hello, [Pork Rasher Error (Packaging)](https://universe.roboflow.com/hello-2aqe0/pork-rasher-error-packaging), Roboflow Universe, **version 4** (reported as CC BY 4.0 by the dataset provider).
- **Format:** RGB images and YOLO object-detection annotations (class ID, normalised box centre, width and height).
- **Labels:** `loose-meat`, `packaging-error`, `twisted-meat`, `unsealed`, `wrinkle`.
- **Images:** 3,349 training, 120 validation and 80 held-out test images, as reported in the executed Colab runs.
- **Labelled bounding boxes:** 2,923 training, 121 validation and 79 test instances. Counts of boxes are not counts of images; some images have no labelled defects.

The dataset itself is **not republished** in this repository. Training images may include augmented versions of original frames. Independence of production events or video frames between dataset splits was not independently verified; see [`DATASET_AND_LIMITATIONS.md`](DATASET_AND_LIMITATIONS.md).

## Computational method

A pretrained Ultralytics `yolov8n.pt` model was fine-tuned on the five annotation classes. YOLOv8n predicts bounding boxes and class labels rather than an image-level category. Two 12-epoch experiments were run in Google Colab on an NVIDIA Tesla T4 GPU, using the same Roboflow v4 splits and primary settings.

| Setting | Baseline | Higher resolution |
|---|---:|---:|
| Input resolution | 416 × 416 | 640 × 640 |
| Epochs | 12 | 12 |
| Batch size | 8 | 8 |
| Random seed | 42 | 42 |
| GPU | Tesla T4 | Tesla T4 |
| Optimiser reported in the runs | AdamW (auto) | AdamW (auto) |

### Validation results

| Metric | 416 px | 640 px |
|---|---:|---:|
| Precision | 0.463 | 0.659 |
| Recall | 0.386 | 0.486 |
| mAP50 | 0.450 | 0.478 |
| mAP50–95 | **0.288** | 0.261 |
| Approximate training time | **7.2 min** | 10.0 min |

The 640-pixel run improved precision, recall and mAP50, but **did not improve mAP50–95** and took longer to train. It was selected for the final test because of its higher validation precision, recall and mAP50; this should not be interpreted as better performance on every metric.

### Held-out test result (selected 640-pixel model)

| Precision | Recall | mAP50 | mAP50–95 | Test images |
|---:|---:|---:|---:|---:|
| 0.5063 | 0.4136 | 0.3553 | 0.1969 | 80 |

Test results differ substantially between classes. The `unsealed` class had the highest observed test mAP50 (0.886), while minority categories had few test annotations; `wrinkle` had none and cannot be assessed on this split. The selected model's test mAP50 was below its validation mAP50, so the results do not support operational deployment.

The figures and tabulated results in this repository come from saved Ultralytics run archives or from transcriptions of the student's executed Colab outputs, as labelled in [`results/README_EVIDENCE.md`](results/README_EVIDENCE.md). They were **not independently reproduced by executing the packaged helper scripts**.

## Repository guide

| Path | Contents |
|---|---|
| [`notebooks/`](notebooks/) | Colab training notebook used as project evidence |
| [`mma3001_pork/`](mma3001_pork/) | Python modules for dataset checks, training, evaluation and prediction |
| [`tests/`](tests/) | Unit tests for supporting code |
| [`results/metrics/`](results/metrics/) | Saved training logs, summary tables and class counts |
| [`results/figures/`](results/figures/) | Saved training curves, PR plots, confusion matrices and example predictions |
| [`results/experiment_metrics.json`](results/experiment_metrics.json) | Recorded model settings and test metrics |
| [`results/pytest_report.txt`](results/pytest_report.txt) | Recorded test output: 8 passed |
| [`docs/`](docs/) | Generated HTML documentation for Python modules |
| [`report/REPORT_OUTLINE.md`](report/REPORT_OUTLINE.md) | Report-writing outline (not the final submitted report) |
| [`PRESENTATION_OUTLINE.md`](PRESENTATION_OUTLINE.md) | Presentation planning notes |
| [`DATASET_AND_LIMITATIONS.md`](DATASET_AND_LIMITATIONS.md) | Dataset usage and known limitations |
| [`requirements.txt`](requirements.txt), [`LICENSE`](LICENSE) | Dependencies and original helper-code licence |

## Set up and run

A suitable Python environment with a GPU is recommended. The recorded Colab environment used Python 3.13 and Ultralytics 8.4.187.

```bash
pip install -r requirements.txt
```

Download **Roboflow version 4** in YOLOv8 format using your own authorised access. Keep authentication secrets private. Once the extracted dataset is available locally, prepare the YOLO YAML with paths for your environment:

```bash
python -m mma3001_pork.dataset --dataset "/content/Pork-Rasher-Error(Packaging)-4" --output /content/mma3001_data.yaml
```

Reproduce the two training configurations (these commands run a new experiment; they do not recreate the exact original checkpoint byte-for-byte):

```bash
python -m mma3001_pork.experiment --data /content/mma3001_data.yaml --variant baseline --runs /content/mma3001_runs
python -m mma3001_pork.experiment --data /content/mma3001_data.yaml --variant highres --runs /content/mma3001_runs
```

Evaluate a **previously selected** 640-pixel checkpoint on the test split without using test results to tune the model:

```bash
python -m mma3001_pork.evaluation --data /content/mma3001_data.yaml --weights /content/mma3001_runs/yolov8n_640/weights/best.pt --split test --imgsz 640 --output results/final_test_rerun.json
```

Generate annotated predictions:

```bash
python -m mma3001_pork.predict --weights /content/mma3001_runs/yolov8n_640/weights/best.pt --source "/content/Pork-Rasher-Error(Packaging)-4/test/images" --output-dir /content/mma3001_predictions
```

The original pretrained weights, fitted checkpoints and licensed image dataset are not included in this repository. Paths above are examples for a Colab environment and must match your extracted files. The helper scripts were prepared as a documented workflow after the student's main Colab experiments; the included notebook and saved run records are the primary evidence for reported results.

## Code verification and documentation

Run the supporting-code tests with:

```bash
python -m pytest -q
```

The saved test record in [`results/pytest_report.txt`](results/pytest_report.txt) reports **8 passed**. These are unit tests of helper functions, **not** independent validation of detection accuracy. The repository also includes generated HTML module documentation in [`docs/`](docs/).

## Limitations

The dataset is class-imbalanced, and `wrinkle` has no labelled examples in either the validation or test split. Related frames may appear across splits, and this was not checked using production-event or recording-session identifiers. Only two image sizes and one 12-epoch run per setting were compared. Results may change with different splits, random seeds, lighting, packaging formats and thresholds. RGB analysis does not establish microbiological quality or physical seal integrity.

## AI use and attribution

ChatGPT assisted with explanation, troubleshooting, Python helper-code scaffolding, repository documentation and report drafting. The main training and evaluation runs were conducted in the student's Colab environment. Reported experimental values were checked against the saved logs and figures; AI-generated code and text still require the student's review. Dataset and pretrained model materials belong to their respective providers and are not claimed as original work.

**Main references:** [Roboflow dataset](https://universe.roboflow.com/hello-2aqe0/pork-rasher-error-packaging), [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/), [Ultralytics software](https://github.com/ultralytics/ultralytics).
