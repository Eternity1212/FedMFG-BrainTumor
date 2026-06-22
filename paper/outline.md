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
| Brisc2025 | pending | 2D | T1 | pending | pending | pending |

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

当前来自历史 JSON 的可用表：

| Algorithm | Accuracy (%) | Macro F1 (%) |
| --- | ---: | ---: |
| FD | 80.79 | 76.62 |
| FedGH | 73.18 | 68.28 |
| FedMM | 78.58 | 77.55 |
| FedProto | 83.95 | 75.93 |
| FedTGP | 87.63 | 87.01 |
| Local | 84.25 | 84.05 |
| FedAMM | pending test rerun | pending test rerun |
| FedMFG | pending test rerun | pending test rerun |

### 4.5 Ablation study

变体：

- Full FedMFG
- w/o modality gate
- w/o combo prototype
- w/o teacher prototype
- w/o prototype alignment loss
- w/o head calibration loss
- uniform head aggregation

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
