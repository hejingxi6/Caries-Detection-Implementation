# Caries-Detection-Implementation

## English

This repository contains the implementation files for a preliminary YOLO-based dental caries detection project.

The project was developed through **multiple controlled optimization rounds** rather than a single one-shot experiment.  
Its purpose was to complete a **basic working implementation**, verify the end-to-end pipeline, and iteratively improve lesion detection performance through practical engineering adjustments.

### Repository Structure

- `yolo_first_round_opt/`  
  First-round optimization package.  
  Focus: establishing a stable training pipeline, running a full validation workflow, and performing threshold sweep analysis.

- `yolo_second_round_opt/`  
  Second-round optimization package.  
  Focus: rebalancing training data exposure to improve minority-lesion recall.

- `yolo_third_round_opt/`  
  Third-round optimization package.  
  Focus: lesion-mix style augmentation, higher input resolution, and further localization-oriented refinement.

- `.gitignore`  
  Excludes cache files, model weights, training outputs, and other unnecessary large or local files.

### Project Goal

The goal of this project is to build a **basic but working YOLO-based detector** for dental caries screening and to improve it through staged experiments.

The work emphasizes:

- reproducible training and evaluation workflow  
- threshold-based validation analysis  
- class imbalance handling  
- small-lesion detection improvement  
- literature-guided interpretation of results  

### Retained Experimental Outcome

According to the retained experimental evidence:

- **Round 2 sweep-best** achieved the strongest **mAP@0.5 = 0.595** and **Recall = 0.592**
- **Round 3 sweep-best** achieved the strongest **mAP@0.5:0.95 = 0.425** and **Precision = 0.677**

These results indicate **clear improvement over the retained initial baseline**, although the detector still has meaningful limitations on minority lesion classes and should not be treated as a final high-performance system.

### Notes

- This repository focuses on implementation files and selected experiment materials.
- Large datasets, model weights, cache files, and bulky training outputs are intentionally excluded.
- The repository is intended for project review, code organization, and technical discussion.
- Detailed quantitative comparison and error analysis are documented separately in the experimental report.

---

## 中文

这个仓库用于保存一个基于 YOLO 的龋齿检测（dental caries detection）初步筛选项目的实现文件。

这个项目不是一次性完成的单轮实验，而是按照**多轮受控优化**的方式推进。  
它的目标是先完成一个**基础可运行的检测实现**，验证完整流程可以跑通，再通过逐轮改进去提升病灶检测效果。

### 仓库结构

- `yolo_first_round_opt/`  
  第一轮优化文件。  
  重点：建立稳定的训练流程，完成完整验证，并进行阈值扫描分析。

- `yolo_second_round_opt/`  
  第二轮优化文件。  
  重点：通过重平衡训练数据暴露，提升少数病灶类别的召回率。

- `yolo_third_round_opt/`  
  第三轮优化文件。  
  重点：通过 lesion-mix 风格增强和更高输入分辨率，进一步改进定位质量。

- `.gitignore`  
  用于排除缓存文件、模型权重、训练输出以及不需要上传的大文件或本地文件。

### 项目目标

本项目的目标是完成一个**基础可运行的 YOLO 龋齿检测器**，并通过分阶段实验逐步改进效果。

项目重点包括：

- 可复现的训练与评估流程  
- 基于阈值扫描的验证分析  
- 类别不平衡处理  
- 小病灶检测能力提升  
- 结合文献进行结果解释  

### 当前保留实验结果

根据目前保留的实验结果：

- **Round 2 的 sweep-best** 取得了最高的 **mAP@0.5 = 0.595** 和 **Recall = 0.592**
- **Round 3 的 sweep-best** 取得了最高的 **mAP@0.5:0.95 = 0.425** 和 **Precision = 0.677**

这些结果说明，相比于初始保留基线，项目取得了**明确的量化提升**；  
但同时也说明，这个检测器对少数病灶类别仍然存在明显局限，**还不能算高性能最终系统**。

### 说明

- 本仓库主要保存实现代码和部分实验材料。
- 大型数据集、模型权重、缓存文件和体积较大的训练输出文件已被有意排除。
- 本仓库主要用于项目整理、代码审查和后续技术讨论。
- 更详细的数值对比和误差分析已另外整理在实验报告中。
