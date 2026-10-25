# Report writing scaffold — 5–10 pages, excluding references and appendices

**Use student-authored reasoning and actual plots from Google Colab. This is an OUTLINE, not a completed submission.** The recorded values below came from user-provided logs and must be verified against exported results.

## 1. Introduction & engineering problem (0.75 page)
- Industrial motivation: faster preliminary visual inspection of pork rasher packages.
- Clear scope: identify **five annotated visible defect categories** in RGB images, with object bounding boxes, not certified food safety testing.
- Problem statement, intended deployment as decision support, risks of missed defects.

## 2. Inputs, outputs, dataset (1 page)
- Images: RGB, YOLOv8 annotation file lines `class x_center y_center width height`, normalised values.
- Classes: loose-meat, packaging-error, twisted-meat, unsealed, wrinkle.
- Counts: 3349 train, 120 validation, 80 test images; cite dataset owner/link/licence.
- Provide **per-class instance distribution plot/table**. Show wrinkle has zero validation/test labels; note train augmentations/near-duplicate risk.
- Invalid input handling: corrupted/non-RGB/unlabelled, unsupported conditions; box ranges and file validation in source module.

## 3. Computational solution (1–1.5 pages)
- Pretrained YOLOv8n, transfer learning, one-stage detector output (class, confidence, box), PyTorch and GPU T4.
- Pipeline: download labelled dataset → train → monitor validation → select `best.pt` → evaluate held-out test.
- Parameters: seed 42, 12 epochs, batch 8, sizes 416 and 640; same splits; no claim of convergence.
- Why choose YOLOv8n: feasible runtime, pretrained detection architecture, small compute footprint. Record software/hardware versions.

## 4. Verification and validation (1–1.5 pages)
- Unit tests verify *software* (annotation parsing, config generation), not ML prediction accuracy.
- Validation: 120 images, used for model selection; quantitative P/R/mAP50/mAP50-95 and optional confusion matrix from original Colab `plots=True`.
- Test: 80 images, 79 defect instances; one independent evaluation after selecting 640 model.
- Explain IoU matching and AP, and that none of these metrics equal simple image classification accuracy.
- Include 2–4 prediction examples (true positive, false positive, missed defect) from saved plots; cite actual images.

## 5. Comparison and optimisation (1 page)
- Baseline 416: validation P=.463, R=.386, mAP50=.450, mAP50-95=.288, 7.2 min training.
- Higher resolution 640: validation P=.659, R=.486, mAP50=.478, mAP50-95=.261, 10.0 min training.
- Changes: +19.6 percentage points precision, +10.0 recall, +2.8 mAP50, -2.7 mAP50-95. Training ~2.8 min longer (~39% more).
- Scientific inference: larger image may preserve detail and may improve some detections, but **no established causal mechanism from two runs**; worse strict-IoU localisation metric; constrained evidence.

## 6. Final test and discussion (1 page)
- Selected 640 model test: P=.5063, R=.4136, mAP50=.3553, mAP50-95=.1969.
- Validation to test difference: mAP50 fell by 12.27 percentage points. This suggests uncertain generalisation and/or differing sample mix, not proven overfitting.
- Test AP50 unsealed .886 but loose-meat .142, packaging-error .280, twisted-meat .113. `wrinkle` cannot be assessed.
- Explain no independent testing of image-level meat safety/quality; sensitivity to lighting, camera angles, class imbalance, sequence overlap.
- Future improvements: more balanced independently labelled images, independent batches/sessions, grouped splitting, multi-seed training, explicit failure analysis.

## 7. AI tools & critical reflection (0.5–0.75 page)
- Specify **actual** tools and roles. E.g., model workflow explanations, code drafts, error troubleshooting, interpretation support, repository packaging; edit to match reality.
- Mention student personally ran training, visually checked outputs, computed metrics, corrected unsupported claims and identified dataset limitations.
- Include at least one AI limitation or wrong assumption that was corrected, e.g. early simplified binary-classification idea was updated to object detection after examining dataset format.
- Responsibility remains with student; viva requires understanding without AI support.

## 8. Conclusion (0.5 page)
- The system is a **prototype**, not production-ready inspection.
- Some success detecting unsealed packages, but average test mAP50 only .3553 and severe minority-class coverage issues.
- Recommend improved representative datasets and more rigorous external validation before engineering deployment.

## Figures to export from actual Colab run
1. Class distribution by split (from label counting).
2. Validation mAP/precision comparison (numbers above; cite student logs).
3. `results.png` training curve for baseline and highres.
4. Confusion matrix / PR curves from saved `plots=True` results (if actually present).
5. Example true positive / false positive / missed detection from actual inference.

## Required repository and references
Include README, functional source, HTML docs, licence, tests/test report, accessible GitHub and ZIP clone. Reference Roboflow dataset, Ultralytics documentation, and any method/metric sources used. Submitted 5–10-page document via Moodle; 5-minute presentation + 3-minute Q&A.
