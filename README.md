# Caries-Detection-Implementation

## English

This repository contains the implementation files for a preliminary dental caries detection project developed through multiple controlled YOLO-based optimization rounds.

The project was not treated as a single one-shot experiment. Instead, it was advanced step by step: first by building a basic working detector, then by improving the training pipeline, validation protocol, class balance strategy, and lesion-focused modeling design. The main purpose was to verify a complete end-to-end workflow and to study how practical engineering changes affect lesion detection performance under limited and imbalanced data conditions.

### Repository Structure

- `yolo_first_round_opt/`  
  First-round optimization package.  
  Focus: establishing a stable baseline training pipeline, running a complete validation workflow, and conducting initial threshold sweep analysis.

- `yolo_second_round_opt/`  
  Second-round optimization package.  
  Focus: rebalancing training data exposure to improve recall on minority lesion classes.

- `yolo_third_round_opt/`  
  Third-round optimization package.  
  Focus: lesion-mix style augmentation, higher input resolution, and further localization-oriented refinement within the original end-to-end detection setting.

- `yolo_fourth_round_opt/`  
  Fourth-round optimization package.  
  Focus: a two-stage ROI-based upgrade.  
  Stage 1 performs binary localization of tooth/lesion regions, and Stage 2 constructs a lesion-only detection setting for `Caries`, `Cavity`, and `Crack`.  
  This round also includes threshold sweep analysis and hard-case ranking for manual review.

- `.gitignore`  
  Excludes cache files, model weights, generated training outputs, and other unnecessary large or local files.

### Project Goal

The goal of this project is to build a basic but working YOLO-based detector for dental caries screening and then improve it through staged experiments.

The work emphasizes:

- reproducible training and evaluation workflow  
- threshold-based validation analysis  
- handling of class imbalance  
- improvement of small-lesion detection  
- practical error diagnosis based on retained results  
- interpretation guided by real experimental evidence rather than isolated headline metrics  

### Retained Experimental Outcomes

According to the retained experimental evidence:

- **Round 2 sweep-best** achieved the strongest  
  **mAP@0.5 = 0.595** and **Recall = 0.592**

- **Round 3 sweep-best** achieved the strongest overall result in the original end-to-end setting, with  
  **Precision = 0.677**, **Recall = 0.491**, **mAP@0.5 = 0.577**, and **mAP@0.5:0.95 = 0.425**

- **Round 4 two-stage sweep-best** achieved  
  **Precision = 0.557**, **Recall = 0.406**, **mAP@0.5 = 0.498**, and **mAP@0.5:0.95 = 0.315**  
  under the lesion-only Stage-2 validation setting

### Interpretation of the Current Status

The retained evidence suggests the following:

- The project achieved clear quantitative improvement over the retained initial baseline through multiple controlled rounds.
- The upgraded **Round 3** configuration remains the strongest overall result under the original full detection setting.
- The **Round 4** two-stage ROI-based route is still meaningful because it verifies that a lesion-focused redesign can improve parts of the pipeline and provides a clearer basis for error diagnosis.
- In particular, the Round 4 lesion-only setting showed stronger performance on some lesion categories, but it did **not** fully surpass the strongest Round 3 overall result.
- The main remaining bottlenecks are:
  - lesion-versus-background confusion
  - severe class imbalance
  - unstable performance on the hardest minority lesion class, especially `Crack`

### Notes

- This repository focuses on implementation files and selected project materials.
- Large datasets, model weights, cache files, intermediate outputs, and bulky training artifacts are intentionally excluded.
- The repository is intended for project review, code organization, and technical discussion.
- Detailed quantitative comparison, round-to-round diagnosis, and error analysis are documented separately in the project report.

---

## 中文

这个仓库用于保存一个基于 YOLO 的龋齿检测（dental caries detection）初步项目的实现文件。  
整个项目不是一次性完成的单轮实验，而是按照**多轮受控优化**的方式逐步推进。

项目的基本思路是：先完成一个**基础可运行的检测实现**，确认端到端流程能够跑通；然后再围绕训练流程、验证方式、类别不平衡、病灶聚焦设计等方面做逐轮改进，观察这些工程调整对病灶检测效果的实际影响。

### 仓库结构

- `yolo_first_round_opt/`  
  第一轮优化文件。  
  重点：建立稳定的基线训练流程，完成完整验证，并进行初步阈值扫描分析。

- `yolo_second_round_opt/`  
  第二轮优化文件。  
  重点：通过重平衡训练数据暴露，提升少数病灶类别的召回率。

- `yolo_third_round_opt/`  
  第三轮优化文件。  
  重点：在原始端到端检测设定下，通过 lesion-mix 风格增强和更高输入分辨率进一步改进定位质量。

- `yolo_fourth_round_opt/`  
  第四轮优化文件。  
  重点：尝试双阶段 ROI 路线。  
  第一阶段先做 tooth/lesion 区域的二分类定位，第二阶段再构建仅包含 `Caries`、`Cavity`、`Crack` 的 lesion-only 检测设定。  
  这一轮同时包含阈值扫描和 hard-case 排序，用于人工复查和误差诊断。

- `.gitignore`  
  用于排除缓存文件、模型权重、训练输出以及不需要上传的大文件或本地文件。

### 项目目标

本项目的目标是完成一个**基础可运行的 YOLO 龋齿检测器**，并通过分阶段实验逐步改进效果。

项目重点包括：

- 可复现的训练与评估流程  
- 基于阈值扫描的验证分析  
- 类别不平衡处理  
- 小病灶检测能力提升  
- 基于真实实验结果的误差诊断  
- 不只看 headline 指标，而是结合实验过程做解释  

### 当前保留实验结果

根据目前保留的实验结果：

- **Round 2 的 sweep-best** 取得了最高的  
  **mAP@0.5 = 0.595** 和 **Recall = 0.592**

- **Round 3 的 sweep-best** 在原始端到端检测设定下取得了当前最强整体结果：  
  **Precision = 0.677**，**Recall = 0.491**，**mAP@0.5 = 0.577**，**mAP@0.5:0.95 = 0.425**

- **Round 4 双阶段方案的 sweep-best** 在 lesion-only 的 Stage-2 验证设定下取得了：  
  **Precision = 0.557**，**Recall = 0.406**，**mAP@0.5 = 0.498**，**mAP@0.5:0.95 = 0.315**

### 当前结果解释

根据目前保留的实验记录，可以得到以下结论：

- 相比初始保留基线，项目经过多轮受控优化后已经取得了明确的量化提升。
- 在原始完整检测设定下，**第三轮升级版** 仍然是当前整体最强结果。
- **第四轮双阶段 ROI 路线** 仍然有研究价值，因为它验证了 lesion-focused 的任务重构是可行的，并且让误差分析更清晰。
- 但从当前结果来看，第四轮并**没有**在整体上完全超过第三轮最强结果。
- 当前最主要的瓶颈仍然是：
  - lesion 与 background 的混淆
  - 严重的类别不平衡
  - hardest minority class（尤其是 `Crack`）的不稳定表现

### 说明

- 本仓库主要保存实现代码和部分项目材料。
- 大型数据集、模型权重、缓存文件、中间输出和体积较大的训练结果已被有意排除。
- 本仓库主要用于项目整理、代码审查和后续技术讨论。
- 更详细的数值对比、轮次诊断和误差分析已另外整理在项目报告中。
