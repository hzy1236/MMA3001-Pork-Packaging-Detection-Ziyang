# MMA3001 — Pork Rasher Packaging Defect Detection Using RGB Images

**Individual numerical methods and machine learning project | Object detection**

## Aim / 项目目标
Develop and evaluate an RGB-image-based **visible-defect detector** for pork rasher packages, using YOLOv8n and a Roboflow object-detection dataset. Predict class-labelled bounding boxes for packaging/appearance errors, and evaluate generalisation on a held-out test set. **This project does not verify food safety or seal integrity.**

## Dataset / 数据集
- Source: **Hello, Pork Rasher Error (Packaging)**, Roboflow Universe, version 4.
- URL: https://universe.roboflow.com/hello-2aqe0/pork-rasher-error-packaging
- Image format: RGB, converted to YOLOv8 detection annotations (class, normalised box centre, width, height).
- Classes: `loose-meat`, `packaging-error`, `twisted-meat`, `unsealed`, `wrinkle`.
- Splits observed: 3,349 train images, 120 validation, 80 test. Training images may include augmented derivatives; original/sequence independence is **not established**.
- The dataset is **not included** in this code repository. Follow the owner's CC BY 4.0 attribution and permitted use; see `DATASET_AND_LIMITATIONS.md`.

## Model comparison / 模型比较
Two fine-tuning experiments on the *same dataset split*, using pretrained `yolov8n.pt`:

| Experiment | Epochs | Input size | Batch | GPU | Validation P | Validation R | Validation mAP50 | Validation mAP50-95 |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| Baseline | 12 | 416 | 8 | Tesla T4 | .463 | .386 | .450 | .288 |
| Higher-resolution | 12 | 640 | 8 | Tesla T4 | .659 | .486 | .478 | .261 |

The 640-pixel model was evaluated **once on the held-out test set** after validation-based selection:

| Test P | Test R | Test mAP50 | Test mAP50-95 | Test images |
|---:|---:|---:|---:|---:|
| .5063 | .4136 | .3553 | .1969 | 80 |

These are **transcriptions of executed Google Colab logs supplied by the student**, not independently rerun in this repository. See `results/experiment_metrics.json` for recorded details. The 640-pixel model had better validation P/R/mAP50, but **worse** validation mAP50-95 and higher runtime. Low performance and minority-class scarcity rule out production deployment.

## Repository structure
```
.
├── mma3001_pork/              # Documented source modules
│   ├── dataset.py              # Dataset config and annotation audit
│   ├── experiment.py           # Reproduce the two train configurations
│   └── evaluation.py           # Independent validation/test evaluation
│   └── predict.py              # Visualise predicted defect boxes
├── tests/                      # Unit tests on synthetic annotations
├── results/experiment_metrics.json
├── docs/                       # HTML Python module documentation
├── report/REPORT_OUTLINE.md    # Write-up scaffold; check conclusions
├── PRESENTATION_OUTLINE.md     # Five-minute talk and Q&A
├── DATASET_AND_LIMITATIONS.md
├── requirements.txt
└── LICENSE
```

## Installation / 环境
This project was run in Google Colab with Python 3.13, Ultralytics 8.4.187, and NVIDIA Tesla T4. Install Python dependencies in a suitable environment:
```bash
pip install -r requirements.txt
```

## Reproduce / 复现实验
Download Roboflow **version 4**, choosing **YOLOv8** format. You may use an existing `data.yaml` from the extracted dataset. In Colab, an example download is:
```python
!pip install roboflow
import roboflow
roboflow.login()  # private authentication, never include tokens in GitHub
rf = roboflow.Roboflow()
ds = rf.workspace('hello-2aqe0').project('pork-rasher-error-packaging').version(4).download('yolov8')
print(ds.location)
```
Create a machine-local, absolute-path YAML config (path must point to your dataset):
```bash
python -m mma3001_pork.dataset --dataset "/content/Pork-Rasher-Error(Packaging)-4" --output /content/mma3001_data.yaml
```
To *rerun* training (unnecessary if you already have saved checkpoints):
```bash
python -m mma3001_pork.experiment --data /content/mma3001_data.yaml --variant baseline --runs /content/mma3001_runs
python -m mma3001_pork.experiment --data /content/mma3001_data.yaml --variant highres --runs /content/mma3001_runs
```
For annotated prediction images (for the report/demo):
```bash
python -m mma3001_pork.predict --weights /content/mma3001_runs/yolov8n_640/weights/best.pt --source "/content/Pork-Rasher-Error(Packaging)-4/test/images" --output-dir /content/mma3001_predictions
```
For held-out test evaluation of the **previously selected model**:
```bash
python -m mma3001_pork.evaluation --data /content/mma3001_data.yaml --weights /content/mma3001_runs/yolov8n_640/weights/best.pt --split test --imgsz 640 --output results/final_test_rerun.json
```
Avoid tuning any training choices on test results. `experiment.py` performs training and validation only; testing is separate. Check the dataset for overlapping video frames across splits before asserting independent generalisation.

## Test functions and HTML documentation
```bash
python -m pytest -q
python -m pydoc -w mma3001_pork.dataset mma3001_pork.experiment mma3001_pork.evaluation mma3001_pork.predict
```
The second command generates four HTML files in your working directory; place them in `docs/`. This provided ZIP already includes generated HTML docs and a recorded Python test-output file. Unit tests verify configuration/annotation-handling logic, **not model accuracy**. For stricter code docs you can also use `pdoc` or Sphinx.

## Required GitHub submission / 提交
1. Replace author placeholder in `LICENSE`. Add your **real** Colab `.ipynb` via **File → Download → Download .ipynb**; do not claim the included Python scripts were the code actually used during training unless you verify and run them.
2. Add original training outputs (e.g. `results.csv`, validation plots, test plots) and your own screenshots. Saved runs are in `/content/drive/MyDrive/MMA3001/*.zip` from your Colab session. The ZIP kit **does not contain your Google Drive models or outputs**.
3. Create a GitHub repository; upload this source + notebook + generated evidence, commit with meaningful messages; never upload Roboflow tokens, private files, or a dataset without following its licence.
4. Ensure README, source docstrings, HTML docs, tests, test report, licence and a functional main branch; submit accessible repo URL and a **ZIP clone** via Moodle.
5. Submit a **separate 5–10 page report through Moodle** and deliver **5-minute presentation + 3-minute Q&A**. Confirm deadline using unit overview spreadsheet.

## AI use disclosure / AI 使用声明
This repository template, Python modules and report outline were drafted with AI assistance after the student completed the Colab experiments. The student must inspect, run and verify this code, edit claims against original logs, and fully disclose AI's actual contribution. **Do not claim tests or code authored by AI were independently written or verified if they were not.**
