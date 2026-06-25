# FedMFG 论文初稿框架

## Title

FedMFG: Modality-aware Fusion and Prototype-guided Head Aggregation for Heterogeneous Federated Brain Tumor Classification with Missing MRI Modalities

## Abstract Draft

Multi-center brain tumor diagnosis requires models that can learn from distributed MRI data without sharing raw patient images. However, real clinical datasets are heterogeneous: institutions may own different MRI modalities, use either 2D slices or 3D volumes, and cover different class distributions. Existing federated learning methods mainly assume homogeneous model architectures or focus on missing-modality segmentation, making them less suitable for heterogeneous brain tumor classification. We propose FedMFG, a federated framework that combines modality-aware feature fusion, modality-combination class prototypes, and prototype-guided classifier-head aggregation. Each client extracts and fuses only its available modalities, avoiding unreliable zero filling or image synthesis. The server maintains modality-combination-level class prototypes as lightweight semantic anchors and uses prototype consistency to weight classifier-head aggregation across heterogeneous clients. Experiments on a multi-source brain tumor MRI classification setting will evaluate FedMFG against local training, FedGH, FedProto, FedTGP, FD, FedMM, and FedAMM.

## 1. Introduction

需要写清楚：

- 脑肿瘤 MRI 分类的临床意义。
- 多中心协作为什么需要联邦学习。
- 真实多中心数据为什么不整齐：模态缺失、2D/3D 并存、类别分布不一致。
- 现有方法主要不足：
  - 传统 FL 假设模型结构一致。
  - 缺失模态方法常依赖补零、生成或完整模态教师。
  - FedAMM 等接近工作主要面向分割，且不突出分类头可靠聚合。
- 本文贡献。

## 2. Related Work

### 2.1 Missing-modality medical image analysis

参考：

- MMCFormer
- PASSION
- M3AE
- modality-adaptive fusion methods

### 2.2 Federated learning for medical imaging

参考：

- FedAvg
- FedProx
- FedBN
- FedGH
- FedProto
- FedTGP

### 2.3 Federated multimodal learning with missing modalities

重点对比：

- FedAMM
- FedMEMA / FedMEPD
- PmcmFL
- MFCPL
- FIN
- MMiC

## 3. Method

### 3.1 Problem formulation

定义：

- 客户端集合。
- 全局类别集合。
- 全局模态集合。
- 客户端可用模态集合。
- 2D/3D 异构特征提取器。

### 3.2 Modality-aware feature fusion

解释：

- 每个可用模态单独编码。
- 投影到统一维度。
- 使用门控网络对可用模态加权。
- 缺失模态不参与计算。

### 3.3 Modality-combination class prototypes

解释：

- `combo_id` 表示模态组合。
- 原型键为 `(combo_id, class_id)`。
- 聚合时考虑样本数量和与历史全局原型的一致性。

### 3.4 Teacher prototype construction

解释：

- 对少模态组合，使用更完整模态组合的同类原型作为教师信息。
- 通过 `mfg_teacher_lambda` 控制当前组合原型和更完整组合原型的融合。

### 3.5 Prototype-guided classifier-head aggregation

解释：

- 不聚合异构 backbone。
- 只聚合统一输入输出维度的 classifier head。
- 根据原型一致性 `rho` 和模态完整度 `eta` 加权。

### 3.6 Training objective

损失：

- 分类交叉熵。
- 教师原型对齐 MSE。
- 分类头校准损失。

## 4. Experiments

### 4.1 Datasets

必须补表：

| Client | Source | 2D/3D | Modalities | Train | Test | Classes |
| --- | --- | --- | --- | ---: | ---: | --- |
| BraTS | pending | 3D | T1/T1c/T2/FLAIR | pending | pending | pending |
| Shanghai | pending | 3D | T1c/FLAIR | pending | pending | pending |
| Figshare | Hugging Face mirror of Cheng et al. | 2D | T1c | 2522 | 542 | 3 |
| Brisc2025 | Zenodo DOI 10.5281/zenodo.17524350 | 2D | T1 | 5000 | 1000 | 4 |

### 4.2 Baselines

- Local / Solo
- FD
- FedGH
- FedProto
- FedTGP
- FedMM
- FedAMM
- FedMFG

### 4.3 Implementation details

需要记录：

- backbone
- prototype dimension
- optimizer
- learning rate
- batch size
- local epochs
- global rounds
- early stopping
- hardware

### 4.4 Main results

**评估口径说明（重要）**：异构 FL 中各客户端样本量差异极大（2D 客户端样本约占 87%），单纯样本加权会被大客户端主导。因此主指标采用 **客户端宏平均 Macro-F1（Client-Macro F1）**，即先算每个客户端的 Macro-F1 再对客户端求平均，公平反映"对每个机构都好"。同时报告样本加权 Acc 作参考。

#### 4.4.1 CPU 低分辨率预实验（seed42，方向性证据，非最终数字）

> 用途：在 GPU 全分辨率多 seed 结果产出前，作为方法有效性的方向性证据。`±0` 因仅单 seed，不是真实方差。

| Algorithm | SampleW Acc | Client-Macro Acc | **Client-Macro F1** | BraTS(3D难) | Shanghai | Figshare | Brisc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **FedMFG (ours, lowregmid)** | **82.03** | **79.53** | **79.30** | **80.0** | 72.5 | 82.5 | 83.1 |
| FedProto | 81.25 | 78.02 | 77.95 | 70.0 | 80.0 | 73.3 | 88.8 |
| FedMFG (vanilla) | 74.06 | 74.27 | 73.76 | 72.5 | 77.5 | 70.8 | 76.2 |
| FedTGP | 78.44 | 74.24 | 72.71 | 60.0 | 80.0 | 70.4 | 86.6 |
| Local | 76.09 | 72.81 | 71.66 | 60.0 | 77.5 | 75.0 | 78.8 |
| FedAMM | 77.81 | 72.29 | 69.87 | 60.0 | 70.0 | 79.2 | 80.0 |
| FedMM | 78.12 | 71.93 | 69.49 | 50.0 | 77.5 | 79.6 | 80.6 |

要点：调参后的 FedMFG (lowregmid: proto/head/teacher λ=0.05/0.05/0.3, lr=3e-4) 在三个聚合口径同时排名第一，并在最难的 3D 客户端 BraTS 上大幅领先（80.0 vs 次优 72.5）。

#### 4.4.2 GPU 全分辨率多 seed 主表（论文正式数字，待填）

> 由 `experiments/run_gpu_paper.sh`（lr=3e-4 + lowregmid + seed 42/43/44 全分辨率）产出，用 `paper_tools/report_final.py` 汇总 mean±std。

| Algorithm | SampleW Acc | Client-Macro Acc | **Client-Macro F1** | BraTS | Shanghai | Figshare | Brisc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Local | pending | pending | pending | | | | |
| FedGH | pending | pending | pending | | | | |
| FedProto | pending | pending | pending | | | | |
| FedTGP | pending | pending | pending | | | | |
| FedMM | pending | pending | pending | | | | |
| FedAMM | pending | pending | pending | | | | |
| **FedMFG (ours)** | pending | pending | pending | | | | |

### 4.5 Ablation study

变体：

- Full FedMFG
- w/o modality gate
- w/o combo prototype
- w/o teacher prototype
- w/o prototype alignment loss
- w/o head calibration loss
- uniform head aggregation

消融基线为 lowregmid 完整 FedMFG，逐项关闭组件（GPU 脚本 STAGE 4 自动跑：`--mfg_disable_teacher` / `--mfg_disable_combo_prototype` / `--mfg_disable_modality_gate` / `--mfg_head_weight_mode uniform`）。

| Variant | Client-Macro F1 | Δ vs Full | 结论 |
| --- | ---: | ---: | --- |
| **Full FedMFG (lowregmid)** | pending (GPU) | — | 完整方法 |
| w/o teacher prototype | CPU 预实验早停，末轮测试 Acc≈0.70（完整≈0.82），明显下降 | ↓ | 教师原型有贡献 |
| w/o combo prototype | pending | | |
| w/o modality gate | pending | | |
| uniform head aggregation | pending | | |

> CPU 预实验已观察到去掉教师原型后性能下降，初步支持该组件的有效性；正式消融数字以 GPU 全分辨率为准。个性化头方向（`--mfg_head_personal_alpha`>0）经 a03/a07 扫描证伪（性能下降），故最终方法采用 alpha=0 共享全局原型指导头。

### 4.6 Missing-modality robustness

待补：

- 25% missing
- 50% missing
- 75% missing
- single-modality client ratio

### 4.7 Visualization

需要：

- confusion matrix
- t-SNE/UMAP
- prototype similarity heatmap
- modality gate attention

## 5. Discussion

要讨论：

- 为什么 FedMFG 适合 2D/3D 异构。
- 为什么原型比全模型参数更适合跨结构协同。
- 与 FedAMM 的差别和适用场景。
- 局限性：仍需完整多中心数据、真实私有数据合规、计算成本较高。

## 6. Conclusion

总结 FedMFG 对模态缺失异构联邦脑肿瘤分类的贡献。
