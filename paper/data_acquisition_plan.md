# 数据获取结论与替代方案

更新时间：2026-06-22

## 结论概览

原计划四客户端中：

- `Figshare` 已经可以使用，已预处理完成。
- `Brisc2025` 已找到正式公开源，推荐立刻用 Zenodo 正式版替换当前的慢速 Hugging Face 替代数据。
- `BraTS/FeTS` 能找到正式数据源，但需要 Synapse/TCIA 授权，不能保证匿名脚本直接下载。
- `Shanghai` 暂未找到与项目设定完全对应的公开下载源，较可能是私有医院数据；如果拿不到，应替换为 TCIA 公开多模态数据或用 FeTS/BraTS 构造双模态客户端。

## 推荐优先级

第一优先级：补齐可直接公开下载的 2D 客户端。

- `Figshare`: 已完成。
- `BRISC2025`: Zenodo DOI `10.5281/zenodo.17524350`，`brisc2025.zip` 约 260MB，CC BY 4.0。

第二优先级：申请或下载 3D 多模态客户端。

- `FeTS/BraTS`: 最贴合联邦学习，因为有机构划分和 T1/T1c/T2/FLAIR 四模态。
- `UPENN-GBM/UCSF-PDGM/UTSW-Glioma`: 可作为公开 3D 多模态替代，来自 TCIA，通常是 CC BY 4.0。

第三优先级：外部验证或扩展类别。

- `Pretreat-MetsToBrain-Masks`: 200 例脑转移瘤，T1/T1CE/T2/FLAIR，适合 `brain_metastases` 类或外部验证。
- `Yale-Brain-Mets-Longitudinal`: 1430 患者、约 43GB，适合大规模外部验证，但下载成本较高。

## 原始目标数据状态

### BraTS / FeTS

能找到。

主要入口：

- BraTS 2021 Synapse: https://www.synapse.org/Synapse:syn25829067
- BraTS 2021 TCIA: https://www.cancerimagingarchive.net/analysis-result/rsna-asnr-miccai-brats-2021/
- FeTS data page: https://fets-ai.github.io/Challenge/data/
- FeTS Synapse: https://www.synapse.org/Synapse:syn54079892/wiki/626854

判断：

- 适合论文主实验。
- 需要注册、同意数据使用条款或授权。
- 如果授权顺利，应优先用 FeTS 的 institution-wise partition 构造联邦客户端。

### Shanghai

暂时没有找到明确公开源。

判断：

- 很可能是原毕业设计中的私有或半私有医院数据。
- 如果用户不能提供原始数据或预处理后的 `data/processed/Shanghai`，不建议在论文中声称使用 Shanghai 私有数据。
- 替代路线是从 BraTS/FeTS/TCIA 中抽取 T1c + FLAIR 作为“partial-modality 3D client”。

### BRISC2025

已找到正式公开源。

主要入口：

- Kaggle: https://www.kaggle.com/datasets/briscdataset/brisc2025/
- Zenodo: https://doi.org/10.5281/zenodo.17524350
- Figshare: https://doi.org/10.6084/m9.figshare.30533120

判断：

- 可直接作为 `Brisc2025` 客户端。
- 数据包含 6000 张 T1 MRI，5,000 train / 1,000 test。
- 类别包含 glioma、meningioma、pituitary、no_tumor。
- 项目已新增 `download_brisc2025_zenodo.sh` 和 `preprocess_brisc2025.py`。

## 公开替代数据

### 2D 分类数据

可选：

- Mendeley `Brain Tumor MRI Dataset (Glioma, Meningioma, Pituitary, No Tumor)`，约 12,064 张 T1CE 图像。
- Zenodo `20: Brain Tumor MRI Dataset`，DOI `10.5281/zenodo.7786009`，页面说明 7,022 张图像，但 API 中主要是模型和指标文件，不建议作为首选原始数据。
- Hugging Face `Simezu/brain-tumour-MRI-scan`，已验证可读，但单文件流式下载较慢。
- Hugging Face `OctoMed/BrainTumor`，3,264 张图像，带多模态问答字段，适合作为轻量图像分类替代源。

推荐：

- 正式论文优先使用 `BRISC2025 + Figshare`。
- 如果要做更多 2D 客户端，再补 Mendeley 或 Kaggle 同类数据。

### 3D 多模态数据

可选：

- `UPENN-GBM`: 多模态 MRI，NIfTI，CC BY 4.0。
- `UCSF-PDGM`: 约 501 成人弥漫性胶质瘤，NIfTI，CC BY 4.0。
- `UTSW-Glioma`: 625 患者，多模态 MRI + 分割 + 分子标签，CC BY 4.0。
- `Pretreat-MetsToBrain-Masks`: 200 脑转移瘤患者，T1/T1CE/T2/FLAIR。

推荐：

- 如果 `BraTS/FeTS` 授权慢，先选一个 TCIA 公开 3D 数据做多模态客户端。
- 论文写作中要明确这是“公开数据模拟多中心异构联邦场景”。

## 下一步执行顺序

1. 下载并预处理正式 `BRISC2025`。
2. 用 `Figshare + BRISC2025` 跑 2D 双客户端主方法与 baseline smoke/小规模实验。
3. 申请或下载 `FeTS/BraTS`。
4. 如果 `Shanghai` 无法获得，用 `FeTS/BraTS` 或 TCIA 数据构造一个 T1c+FLAIR 3D partial-modality client。
5. 数据齐全后，启动多 seed 正式主实验和消融实验。
