# Five-minute presentation and three-minute Q&A / 5分钟演讲

- **0:00–0:40 — Problem / 问题**: Why visible-defect screening matters, RGB constraints, inputs/output.
- **0:40–1:20 — Dataset / 数据**: Roboflow five classes, train/val/test split, class imbalance and `wrinkle` absence in held-out sets.
- **1:20–2:10 — Method / 方法**: YOLOv8n transfer learning, boxes and confidence, 416 baseline vs 640 variant, show GitHub and docstrings.
- **2:10–3:20 — Validation / 结果**: Compare P/R/mAP validation; show final held-out test P .5063, R .4136, mAP50 .3553, mAP50-95 .1969; one qualitative detection.
- **3:20–4:20 — Evaluation / 反思**: Higher resolution trade-off, class imbalance, unseen/similar video frames risk, real failures, no food-safety claim.
- **4:20–5:00 — GitHub + AI reflection + conclusion**: Reproducible code/docs/tests, AI-assisted planning/code drafts explained honestly, own decisions/verification.

## Example Q&A / 常见问题
**Why YOLOv8n?** Pretrained lightweight detector, feasible on T4 GPU, object boxes correspond to dataset format.

**How did you validate?** 120-image validation split for model selection, 80-image held-out test set used only after selecting 640px; object detection P/R/AP plus dataset audit.

**Why not accuracy?** Detecting boxes requires correct classes AND box overlap. mAP integrates precision–recall at defined IoU conditions; ordinary classification accuracy hides localisation errors.

**What causes low mAP?** Limited minority class annotations, difficult visual distinctions, short training, possibly correlated frames; cannot attribute causality without additional experiments.

**Why did 640 improve some metrics but not all?** More image detail and compute cost; strict box IoU score worsened; evidence supports trade-offs, not universal superiority.

**How do you know code works?** Unit tests for annotations/config; actual model judged separately on labelled withheld samples and visual case inspection. Tests ≠ detection validation.

**Did AI write the code?** Disclose actual use precisely. You ran and reviewed executed experiments, interpreted real outputs; project ownership and verification remain your responsibility.
