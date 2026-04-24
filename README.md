# Transfer Learning for Dental Caries Detection Using YOLOv8

## English

This repository documents a YOLOv8-based dental caries detection project as an applied case study of transfer learning. A pre-trained YOLOv8 object detector was fine-tuned on a small and imbalanced dental image dataset to examine its performance, limitations, and possible improvement directions.

This project is **not** presented as a clinically reliable diagnostic system. Instead, it is used as a machine learning case study to understand how transfer learning behaves in a small-scale medical object detection task.

## Project Positioning

The project was originally developed as a YOLO-based dental caries detection implementation and was later reorganized as a transfer learning case study.

The core idea is:

- start from a pre-trained YOLOv8 object detection model;
- fine-tune it on dental image data;
- evaluate the detector using object detection metrics;
- analyze why performance is limited under small, imbalanced, and visually difficult medical data;
- identify future improvement directions rather than overclaiming clinical reliability.

Because this is an object detection task, the main metrics are:

- Precision
- Recall
- mAP@0.5
- mAP@0.5:0.95

Classification accuracy is not used as the main metric because the model needs to both locate and classify dental regions.

## Repository Structure

```text
Caries-Detection-Implementation/
├── yolo_first_round_opt/
├── yolo_second_round_opt/
├── yolo_third_round_opt/
├── yolo_fourth_round_opt/
├── report/
├── .gitignore
└── README.md
```

### `yolo_first_round_opt/`

First-round optimization package.

Focus:

- establishing a stable YOLOv8 baseline training workflow;
- completing validation;
- performing an initial threshold sweep.

### `yolo_second_round_opt/`

Second-round optimization package.

Focus:

- rebalancing training data exposure;
- improving recall on minority lesion classes;
- selecting a better operating point through threshold sweep.

### `yolo_third_round_opt/`

Third-round optimization package.

Focus:

- lesion-mix style augmentation;
- higher input resolution;
- further localization-oriented refinement under the original full-image detection setting.

Round 3 remains the strongest full-image result in the retained evidence.

### `yolo_fourth_round_opt/`

Fourth-round optimization package.

Focus:

- two-stage ROI-based redesign;
- Stage 1: binary tooth-plus-lesion localization;
- Stage 2: lesion-only detection for `Caries`, `Cavity`, and `Crack`;
- empty-sample filtering;
- threshold calibration;
- hard-case review.

Round 4 should not be interpreted as replacing Round 3 overall. Its value is methodological: it improved Caries and Cavity in a lesion-focused setting and made the remaining bottlenecks more visible.

## Transfer Learning Interpretation

This project follows the transfer learning logic:

1. A YOLOv8 model is initialized from pre-trained weights.
2. The model is fine-tuned on a dental image dataset.
3. The detection outputs are adapted to dental-related classes.
4. The model is evaluated under a limited and imbalanced target domain.

Fine-tuning is more suitable than feature extraction alone in this project because object detection requires adapting both classification behavior and bounding-box localization.

A true random-initialization training-from-scratch comparison was **not** conducted. Instead, the project uses a conceptual comparison:

| Approach | Status | Interpretation |
|---|---|---|
| Training from scratch | Not conducted | Requires more data and compute; likely unstable on a small and imbalanced medical dataset. |
| Feature extraction only | Conceptually possible | Less suitable because object detection requires adapting detection heads and spatial localization. |
| Fine-tuning pre-trained YOLOv8 | Used | Best fit because it transfers general visual detection features and adapts them to dental lesion classes. |

## Retained Experimental Results

The retained experimental evidence is summarized below.

| Variant | Main setting | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---:|---:|---:|---:|
| Initial baseline | Basic YOLOv8 sanity pipeline | 0.331 | 0.362 | 0.312 | 0.216 |
| Round 2 sweep-best | Rebalancing / minority exposure | 0.590 | 0.592 | 0.595 | 0.403 |
| Round 3 sweep-best | Lesion-mix + higher-resolution full-image detector | 0.677 | 0.491 | 0.577 | 0.425 |
| Round 4 sweep-best | Two-stage ROI lesion-focused pipeline | 0.557 | 0.406 | 0.498 | 0.315 |

## Interpretation of Results

Compared with the retained initial baseline, the project achieved measurable improvement.

- Round 2 produced the strongest retained mAP@0.5 and recall.
- Round 3 produced the strongest full-image result, with the highest precision and the best mAP@0.5:0.95.
- Round 4 did not surpass Round 3 overall, but it improved Caries and Cavity under a lesion-focused ROI pipeline.

The correct interpretation is not that the detector is clinically reliable. The better interpretation is that transfer learning can produce a workable detector under limited medical data, while still being constrained by:

- small lesion size;
- lesion-versus-background confusion;
- severe class imbalance;
- unstable performance on the hardest minority class, especially `Crack`.

## Round 3 and Round 4 Relationship

Round 3 remains the main full-image reference point.

Round 4 was a methodological exploration. It moved the project from a single-stage full-image detector to a two-stage ROI-based pipeline. This helped clarify where the bottlenecks are.

Class-wise interpretation:

| Class | Round 3 mAP@0.5 | Round 4 mAP@0.5 | Interpretation |
|---|---:|---:|---|
| Caries | 0.345 | 0.439 | Improved in Round 4 |
| Cavity | 0.397 | 0.529 | Improved in Round 4 |
| Crack | 0.384 | 0.344 | Still weak; hardest minority class |

The Round 4 result should therefore be described as a useful lesion-focused redesign, not as the overall best model.

## Limitations

This project has clear limitations:

- it is not clinical-ready;
- no true from-scratch training baseline was conducted;
- model performance remains limited by small lesions;
- class imbalance remains severe;
- `Crack` is still unstable because it has far fewer examples;
- background confusion remains a major error source;
- the model would require external validation before any real medical use.

## Future Directions

Future work should focus on:

- SAHI / sliced inference for small lesions;
- annotation quality audit;
- hard false-positive and false-negative review;
- targeted strengthening of the `Crack` class;
- domain adaptation from general object detection to dental imagery;
- few-shot learning for rare lesion classes;
- self-supervised pre-training on larger unlabeled dental image collections.

## Privacy and Data Availability

Large datasets, model weights, cache files, and bulky training outputs are intentionally excluded from this public repository.

This repository is intended for code organization, project review, and technical discussion. It should not be treated as a deployed medical system.

---

## 中文

本仓库记录了一个基于 YOLOv8 的龋齿检测项目，并将其整理为一个迁移学习应用案例。项目使用预训练 YOLOv8 目标检测模型，并在小规模、类别不平衡的牙科图像数据集上进行微调，用于分析模型表现、局限性和后续改进方向。

本项目**不声称已经达到临床诊断水平**。它更适合作为一个机器学习案例，用来说明迁移学习在小规模医学目标检测任务中的作用和限制。

## 项目定位

本项目原本是一个 YOLO 龋齿检测技术项目，后来被重新整理为 CDS521 Transfer Learning 方向的案例研究。

核心逻辑是：

- 使用预训练 YOLOv8 模型；
- 在牙科图像数据上进行微调；
- 使用目标检测指标进行评估；
- 分析小规模、类别不平衡和小病灶检测带来的限制；
- 提出合理的后续改进方向，而不是夸大临床可靠性。

因为这是目标检测任务，所以主要指标是：

- Precision
- Recall
- mAP@0.5
- mAP@0.5:0.95

不使用普通 classification accuracy 作为主要指标，因为模型需要同时完成定位和分类。

## 当前主要结论

保留实验结果显示：

- Round 2 在 mAP@0.5 和 recall 上最好；
- Round 3 是当前最强的 full-image 检测结果；
- Round 4 没有整体超过 Round 3，但它通过 two-stage ROI pipeline 改善了 Caries 和 Cavity，并让 Crack 少数类问题更加清楚。

因此，本项目最合理的结论是：

迁移学习可以帮助预训练 YOLOv8 模型在有限牙科数据上形成可用的检测能力，但模型仍然受到小病灶、背景混淆、类别不平衡和 Crack 少数类不稳定的限制。

## 不应过度解读

本项目不应被解释为临床诊断系统。它是一个 transfer learning case study，而不是可以直接部署的医疗 AI 产品。
