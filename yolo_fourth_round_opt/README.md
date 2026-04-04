# yolo_fourth_round_opt

## English

### Overview
This folder contains the **Round 4 upgraded implementation** for the preliminary dental caries detection project.

Round 4 was developed **after the upgraded Round 3 result**. It is **not** another small tuning attempt on the same end-to-end detector. Instead, this round explores a **two-stage ROI-based pipeline**:

- **Stage 1:** binary localization of tooth / lesion foreground
- **Stage 2:** lesion-only detection for `Caries`, `Cavity`, and `Crack`

The purpose of Round 4 is to test whether a lesion-focused redesign can:

- reduce `Tooth` dominance
- improve lesion-category learning
- provide clearer error diagnosis

---

### What Round 4 Changed
Compared with the previous single-stage four-class setting, Round 4 introduced:

- a **two-stage pipeline**
- **global normalization statistics**
- **Stage 2 lesion-only dataset rebuilding**
- **empty-sample filtering**
- **validation threshold sweep**
- **hard-case ranking for manual review**

This means Round 4 is a **methodological upgrade**, not just another YOLO run.

---

### Main Pipeline

#### Stage 1: Binary Localization
A lightweight binary model is trained to identify tooth / lesion foreground regions.

Its outputs are used to generate ROI proposals for the downstream detector.

#### Stage 2: Lesion-Only Detection
Using Stage 1 outputs, a lesion-only dataset is built for:

- `Caries`
- `Cavity`
- `Crack`

The `Tooth` class is removed from the final detector so that training focuses on the real lesion categories.

---

### Repository Structure

#### Core scripts
- `00_check_environment.py`  
  Check Python / PyTorch / CUDA environment.
- `01_compute_global_norm.py`  
  Compute global normalization statistics.
- `02_prepare_stage1_binary_masks.py`  
  Build the Stage 1 binary dataset.
- `03_train_stage1_binary.py`  
  Train the Stage 1 binary localization model.
- `04_predict_stage1_binary.py`  
  Run Stage 1 prediction and generate ROI proposals.
- `05_build_stage2_lesion_dataset.py`  
  Build the Stage 2 lesion-only dataset from Stage 1 outputs.
- `06_train_stage2_yolo.py`  
  Train the Stage 2 YOLO detector.
- `07_sweep_stage2_thresholds.py`  
  Run validation threshold sweep.
- `08_rank_hard_cases_from_stage2.py`  
  Export hard validation cases for manual review.
- `check_stage2_labels.py`  
  Check Stage 2 label statistics and empty-sample ratio.
- `make_stage2_v3.py`  
  Rebuild the filtered Stage 2 v3 dataset.
- `common.py`  
  Shared utilities.
- `tiny_unet.py`  
  Lightweight module used in Stage 1.

#### Configuration / support files
- `data_stage1_fixed.yaml`
- `norm_stats.json`
- `requirements.txt`

#### Batch runners
- `RUN_0_install.bat`
- `RUN_1_env_check.bat`
- `RUN_2_compute_norm.bat`
- `RUN_3_prepare_stage1_binary_masks.bat`
- `RUN_4_train_stage1_binary.bat`
- `RUN_5_predict_stage1_binary.bat`
- `RUN_6_build_stage2_dataset.bat`
- `RUN_7_train_stage2_yolo.bat`
- `RUN_8_sweep_stage2.bat`

These `.bat` files were used for convenient execution in a Windows environment.

---

### Recommended Execution Order
1. `RUN_0_install.bat`
2. `RUN_1_env_check.bat`
3. `RUN_2_compute_norm.bat`
4. `RUN_3_prepare_stage1_binary_masks.bat`
5. `RUN_4_train_stage1_binary.bat`
6. `RUN_5_predict_stage1_binary.bat`
7. `RUN_6_build_stage2_dataset.bat`
8. `RUN_7_train_stage2_yolo.bat`
9. `RUN_8_sweep_stage2.bat`
10. `python 08_rank_hard_cases_from_stage2.py ...`

---

### Intermediate Dataset Diagnosis

#### Initial Stage 2 dataset problem
The first version of the Stage 2 dataset had too many empty samples:

- train empty ratio = **35.4%**
- val empty ratio = **46.9%**
- test empty ratio = **37.5%**

This version was considered too noisy for stable lesion-only detection.

#### Revised Stage 2 dataset (v3)
A revised filtering strategy was then applied:

- keep all non-empty samples
- keep only a controlled number of empty crops as hard negatives

The empty-sample ratio was reduced to:

- train = **13.4%**
- val = **12.9%**
- test = **13.7%**

This filtered dataset was retained for the final Stage 2 detector.

---

### Retained Results

#### Stage 1 best result
The Stage 1 binary model reached approximately:

- **best val_dice = 0.8965**
- **best val_iou = 0.8292**

This was considered stable enough for downstream ROI generation.

#### Stage 2 best checkpoint
The final Stage 2 YOLO detector achieved:

- **Precision = 0.567**
- **Recall = 0.411**
- **mAP@0.5 = 0.438**
- **mAP@0.5:0.95 = 0.241**

Class-wise results:

- **Caries:** P 0.557 | R 0.436 | mAP50 0.439 | mAP50-95 0.212
- **Cavity:** P 0.559 | R 0.484 | mAP50 0.529 | mAP50-95 0.272
- **Crack:** P 0.585 | R 0.312 | mAP50 0.344 | mAP50-95 0.237

#### Stage 2 sweep-best result
After threshold sweep on validation data, the best operating point was:

- **conf = 0.30**
- **iou = 0.45**

Sweep-best metrics:

- **Precision = 0.557**
- **Recall = 0.406**
- **mAP@0.5 = 0.498**
- **mAP@0.5:0.95 = 0.315**

This shows that Round 4 is sensitive to threshold choice and performs better after operating-point calibration than under the default checkpoint reading alone.

---

### Comparison with Round 3

#### Round 3 remained stronger overall
Retained Round 3 best checkpoint:

- **Precision = 0.613**
- **Recall = 0.545**
- **mAP@0.5 = 0.523**
- **mAP@0.5:0.95 = 0.365**

Retained Round 3 sweep-best:

- **Precision = 0.677**
- **Recall = 0.491**
- **mAP@0.5 = 0.577**
- **mAP@0.5:0.95 = 0.425**

So the strict conclusion is:

- **Round 3 remains the strongest overall benchmark**
- **Round 4 did not surpass Round 3 in headline metrics**

#### What Round 4 actually contributed
Round 4 still matters because it showed that:

- a two-stage ROI-based route is feasible
- `Caries` and `Cavity` improved in the lesion-focused setting
- the current bottlenecks are clearer:
  - lesion-versus-background confusion
  - severe class imbalance
  - weak `Crack` stability

So the value of Round 4 is **not overall replacement**, but **better methodological understanding and cleaner lesion-focused diagnosis**.

---

### Main Conclusion
The final interpretation of Round 4 is:

- it is **not** a superficial extra run
- it changes the task from a single-stage four-class detector to a two-stage lesion-focused pipeline
- it validates Stage 1 localization
- it rebuilds the Stage 2 dataset
- it reduces the empty-sample problem
- it completes threshold sweep and hard-case review

The strict conclusion is:

- **Round 3 remains the stronger overall benchmark**
- **Round 4 is a meaningful upgrade attempt**
- **Caries and Cavity benefited**
- **Crack remains the hardest minority lesion class**

---

### Notes
- This folder keeps the implementation side of Round 4.
- Large datasets, model weights, caches, and bulky generated artifacts are intentionally excluded.
- Detailed round-to-round comparison is documented separately in the project report.
- This folder is intended for code review, project reporting, and technical inspection of the Round 4 upgrade.

---

## 中文

### 项目概述
这个文件夹保存的是龋齿检测项目的 **Round 4（第四轮）升级实现**。

第四轮是在第三轮升级版结果基础上继续推进的。它**不是**在原有端到端四分类检测器上再做一次小幅调参，而是尝试了一个 **双阶段 ROI 路线**：

- **Stage 1：** tooth / lesion 前景二分类定位
- **Stage 2：** 只对 `Caries`、`Cavity`、`Crack` 做 lesion-only 检测

第四轮的核心目标是测试：

- 能否减弱 `Tooth` 类主导问题
- 能否提升 lesion 类学习效果
- 能否让误差诊断更清晰

---

### 第四轮新增了什么
相比之前的单阶段四分类设定，第四轮新增了：

- **双阶段 pipeline**
- **全局归一化统计**
- **Stage 2 lesion-only 数据集重建**
- **空样本过滤**
- **validation threshold sweep**
- **hard-case 排序与人工复查**

因此，第四轮属于一次**方法级升级**，而不是简单再跑一版 YOLO。

---

### 主要流程

#### Stage 1：二分类定位
先训练一个轻量模型，对 tooth / lesion 前景区域进行定位。

这一阶段的输出会被用来生成后续检测所需的 ROI。

#### Stage 2：Lesion-Only Detection
基于 Stage 1 的输出，构建只包含以下三类的检测数据集：

- `Caries`
- `Cavity`
- `Crack`

最终检测器不再直接学习 `Tooth`，而是更聚焦真正的 lesion 类别。

---

### 仓库结构

#### 核心脚本
- `00_check_environment.py`  
  检查 Python / PyTorch / CUDA 环境
- `01_compute_global_norm.py`  
  计算全局归一化统计量
- `02_prepare_stage1_binary_masks.py`  
  构建 Stage 1 二分类数据
- `03_train_stage1_binary.py`  
  训练 Stage 1 二分类定位模型
- `04_predict_stage1_binary.py`  
  生成 Stage 1 ROI 预测结果
- `05_build_stage2_lesion_dataset.py`  
  基于 Stage 1 输出构建 Stage 2 lesion-only 数据集
- `06_train_stage2_yolo.py`  
  训练 Stage 2 YOLO 检测器
- `07_sweep_stage2_thresholds.py`  
  在 validation 上做阈值扫描
- `08_rank_hard_cases_from_stage2.py`  
  导出难例，供人工复查
- `check_stage2_labels.py`  
  检查 Stage 2 标签统计与空样本比例
- `make_stage2_v3.py`  
  重建过滤后的 Stage 2 v3 数据集
- `common.py`  
  公共工具函数
- `tiny_unet.py`  
  Stage 1 使用的轻量网络模块

#### 配置与辅助文件
- `data_stage1_fixed.yaml`
- `norm_stats.json`
- `requirements.txt`

#### Windows 批处理执行文件
- `RUN_0_install.bat`
- `RUN_1_env_check.bat`
- `RUN_2_compute_norm.bat`
- `RUN_3_prepare_stage1_binary_masks.bat`
- `RUN_4_train_stage1_binary.bat`
- `RUN_5_predict_stage1_binary.bat`
- `RUN_6_build_stage2_dataset.bat`
- `RUN_7_train_stage2_yolo.bat`
- `RUN_8_sweep_stage2.bat`

这些 `.bat` 文件用于在 Windows 环境下顺序执行第四轮流程。

---

### 推荐执行顺序
1. `RUN_0_install.bat`
2. `RUN_1_env_check.bat`
3. `RUN_2_compute_norm.bat`
4. `RUN_3_prepare_stage1_binary_masks.bat`
5. `RUN_4_train_stage1_binary.bat`
6. `RUN_5_predict_stage1_binary.bat`
7. `RUN_6_build_stage2_dataset.bat`
8. `RUN_7_train_stage2_yolo.bat`
9. `RUN_8_sweep_stage2.bat`
10. `python 08_rank_hard_cases_from_stage2.py ...`

---

### 中间数据诊断

#### 初始 Stage 2 数据问题
最初版本的 Stage 2 数据集空样本比例过高：

- train empty ratio = **35.4%**
- val empty ratio = **46.9%**
- test empty ratio = **37.5%**

这个版本噪声过大，不适合做稳定的 lesion-only 检测。

#### 修订后的 Stage 2 数据集（v3）
随后采用了新的过滤策略：

- 保留全部 non-empty 样本
- 只保留少量 empty crop 作为 hard negatives

最终空样本比例下降到：

- train = **13.4%**
- val = **12.9%**
- test = **13.7%**

这个过滤后的版本被保留用于最终的 Stage 2 检测训练。

---

### 保留结果

#### Stage 1 最佳结果
Stage 1 二分类定位模型大致达到：

- **best val_dice = 0.8965**
- **best val_iou = 0.8292**

说明 Stage 1 已经足够稳定，可以支撑后续 ROI 构建。

#### Stage 2 best checkpoint
最终 Stage 2 YOLO 检测器结果为：

- **Precision = 0.567**
- **Recall = 0.411**
- **mAP@0.5 = 0.438**
- **mAP@0.5:0.95 = 0.241**

分类别结果：

- **Caries:** P 0.557 | R 0.436 | mAP50 0.439 | mAP50-95 0.212
- **Cavity:** P 0.559 | R 0.484 | mAP50 0.529 | mAP50-95 0.272
- **Crack:** P 0.585 | R 0.312 | mAP50 0.344 | mAP50-95 0.237

#### Stage 2 sweep-best
在 validation 上进行 threshold sweep 后，最佳 operating point 为：

- **conf = 0.30**
- **iou = 0.45**

对应结果：

- **Precision = 0.557**
- **Recall = 0.406**
- **mAP@0.5 = 0.498**
- **mAP@0.5:0.95 = 0.315**

这说明第四轮模型对阈值较敏感，经过 operating-point 校准后，其能力比默认 best.pt 读数更完整。

---

### 与第三轮的关系

#### 第三轮整体仍然更强
第三轮保留 best.pt：

- **Precision = 0.613**
- **Recall = 0.545**
- **mAP@0.5 = 0.523**
- **mAP@0.5:0.95 = 0.365**

第三轮保留 sweep-best：

- **Precision = 0.677**
- **Recall = 0.491**
- **mAP@0.5 = 0.577**
- **mAP@0.5:0.95 = 0.425**

因此严格结论是：

- **第三轮仍然是当前整体最强 benchmark**
- **第四轮没有在 headline metrics 上超过第三轮**

#### 第四轮真正的价值
第四轮仍然有意义，因为它证明了：

- 双阶段 ROI 路线是可行的
- `Caries` 和 `Cavity` 在 lesion-focused 设定下得到提升
- 当前主要瓶颈更清楚了：
  - lesion 与 background 混淆
  - 严重类别不平衡
  - `Crack` 稳定性仍然偏弱

所以第四轮的价值**不是整体替代第三轮**，而是：

**提供了更清晰的方法判断和更干净的病灶级误差分析基础。**

---

### 主要结论
第四轮最终应被解释为：

- 它**不是**一次表面的额外实验
- 它把任务从单阶段四分类检测器改成了双阶段 lesion-focused pipeline
- 它验证了 Stage 1 定位可行
- 它重建了 Stage 2 数据集
- 它降低了空样本问题
- 它完成了 threshold sweep 和 hard-case review

严格结论是：

- **第三轮仍是更强的整体 benchmark**
- **第四轮是一次有意义的升级尝试**
- **Caries 和 Cavity 得到改善**
- **Crack 仍然是最难的少数类**

---

### 说明
- 本文件夹主要保存第四轮实现代码
- 大型数据集、模型权重、缓存文件及体积较大的自动生成产物未纳入仓库
- 更详细的轮次对比见项目报告
- 本文件夹主要用于代码审查、项目汇报和第四轮升级方案的技术说明
