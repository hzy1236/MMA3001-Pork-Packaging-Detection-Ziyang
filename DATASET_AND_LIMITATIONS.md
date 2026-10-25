# Dataset Provenance, Scope, and Limitations

- Source: Roboflow Universe — Hello, **Pork Rasher Error (Packaging)**, version 4, https://universe.roboflow.com/hello-2aqe0/pork-rasher-error-packaging
- Dataset licence displayed in the student screenshot: **CC BY 4.0**. Attribute source and owner. Verify licence on source page before redistribution.
- Task: RGB object detection of five visible-defect labels, NOT chemical or microbial quality assurance.
- Input: standard RGB image, typically JPG/PNG. Box annotation format: `class_id x_center y_center width height` with normalised coordinates in `[0, 1]`.
- Output: predicted class, confidence score, pixel bounding box; evaluation P/R/AP and visualisations.
- Unsupported: thermal/spectral analysis, unseen camera settings, actual vacuum integrity, internal quality, proof of food safety. Do not infer any of these from an RGB image alone.

## Label-instance distribution observed in Colab

| Label | Train | Validation | Test |
|---|---:|---:|---:|
| loose-meat | 66 | 3 | 3 |
| packaging-error | 1712 | 34 | 23 |
| twisted-meat | 72 | 3 | 4 |
| unsealed | 1055 | 81 | 49 |
| wrinkle | 18 | 0 | 0 |

These are **bounding-box instances**, not image counts. Some images contain multiple annotations and some contain none.

## Threats to validity
1. Severe class imbalance. `wrinkle` has **no validation or test annotations**, so its detection ability cannot be assessed independently.
2. Only 3 or 4 held-out instances for certain classes; per-class metrics may be unstable.
3. Source screenshots appear visually similar/sequence-like. Confirm no near-duplicate original video frames across splits; otherwise measured generalisation can be inflated. **This potential leakage was not audited in the provided results.**
4. Model trained for only 12 epochs. Loss decreases toward the end; convergence was not established.
5. Pretrained weights and default data augmentations were used. Higher resolution changes compute cost, and one seed/run cannot establish robust performance improvements.
6. Held-out test precision/recall/mAP declined relative to validation. This difference could reflect sampling, imbalance, or domain differences; no causal explanation was established.
7. Detection is not the same as image-level quality verdict or a comprehensive meat-quality report. The current prototype detects visible labelled defects only.

## Test results recorded
- Model: YOLOv8n, 640×640, best validation checkpoint.
- Independent held-out test: 80 images, 79 labelled instances, 12 background images, zero corrupt files.
- Test: Precision 0.5063, Recall 0.4136, mAP50 0.3553, mAP50-95 0.1969.
- Inference logs: ~10.2 ms preprocessing, ~8.0 ms model inference, ~7.0 ms postprocessing per image on a Tesla T4; run-specific timings.
- Per class on test: loose-meat P .237, R .237, AP50 .142; packaging-error P .237, R .478, AP50 .280; twisted-meat P 1.0, R 0, AP50 .113; unsealed P .551, R .939, AP50 .886. Interpret tiny classes cautiously. Per-class Precision=1 with Recall=0 does **not** prove perfect detection.
