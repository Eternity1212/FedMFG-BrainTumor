# 数据目录说明

本目录用于后续补实验。真实医学影像数据通常体积较大，且可能有数据使用协议，所以原始数据和预处理后的 `.npz` 文件不会上传到 GitHub。

## 目录结构

```text
data/
  raw/          # 原始下载数据，例如 BraTS NIfTI、Figshare MAT/PNG
  processed/    # 转换成项目代码可读取的 npz 结构
  external/     # 第三方论文代码或参考实现，仅本地学习使用
  scripts/      # 下载、检查和预处理脚本
```

## 项目代码需要的数据结构

训练代码默认读取如下结构：

```text
data/processed/
  BraTS/
    train/
      glioma/
        sample_id/
          t1.npz
          t1c.npz
          t2w.npz
          t2f.npz
    test/
  Shanghai/
  Figshare/
  Brisc2025/
```

每个 `.npz` 文件中需要包含键 `x`：

```python
import numpy as np
np.savez_compressed("t1c.npz", x=image_array.astype("float32"))
```

## 推荐数据来源

### 1. BraTS / FeTS

用途：完整 3D 多模态 MRI，适合作为全模态客户端或联邦真实机构划分参考。

特点：

- 模态一般包括 T1、T1ce/T1c、T2、FLAIR。
- 通常需要 Kaggle、Synapse、CBICA 或 FeTS 平台账号授权。
- FeTS 提供更接近真实联邦场景的 institution-wise 划分。

注意：这类数据不建议直接脚本匿名下载，优先走官方授权流程。

可用路线：

- BraTS 2021: https://www.synapse.org/Synapse:syn25829067
- TCIA BraTS 2021 页面: https://www.cancerimagingarchive.net/analysis-result/rsna-asnr-miccai-brats-2021/
- FeTS Challenge data: https://fets-ai.github.io/Challenge/data/
- FeTS Synapse challenge: https://www.synapse.org/Synapse:syn54079892/wiki/626854

投稿优先级：如果能完成 Synapse/TCIA 授权，优先使用 FeTS/BraTS，因为它提供 T1、T1c、T2、FLAIR 四模态，并且 FeTS 有机构划分，最贴合联邦学习设定。

### 2. Figshare Brain Tumor Dataset

用途：2D T1-CE 脑肿瘤分类客户端。

公开链接：

- Figshare DOI: https://doi.org/10.6084/m9.figshare.1512427.v5
- Hugging Face 镜像: https://huggingface.co/datasets/Angelou0516/Figshare_Brain_Tumor

常见类别：

- meningioma
- glioma
- pituitary

### 3. Br35H / BRISC / Brisc2025 类 2D MRI 数据

用途：补充 2D 单模态客户端，模拟真实多中心分类数据差异。

公开可用来源：

- BRISC2025: https://www.kaggle.com/datasets/briscdataset/brisc2025/
- BRISC Figshare DOI: https://doi.org/10.6084/m9.figshare.30533120
- BRISC Zenodo DOI: https://doi.org/10.5281/zenodo.17524350
- Hugging Face 替代公开集: https://huggingface.co/datasets/Simezu/brain-tumour-MRI-scan

注意：需要确认具体数据来源、类别定义和许可证，避免把不同任务标签混用。`Simezu/brain-tumour-MRI-scan` 是 Figshare、SARTAJ、Br35H 的组合公开集，适合先做公开可复现实验；如果拿到正式 BRISC2025 原始包，应优先替换成正式 BRISC2025。

当前优先级：正式 `BRISC2025` 已确认可从 Zenodo 直接下载 `brisc2025.zip`，大小约 260MB，许可证为 CC BY 4.0，适合直接作为 `Brisc2025` 客户端。

### 4. Shanghai / Yale 私有或半公开数据

用途：作为双模态或外部验证客户端。

注意：如果涉及私有医学数据，论文中必须说明伦理审批、脱敏方式、数据不可公开原因。

当前检索结论：

- `Shanghai` 暂未找到与项目中 `T1c + FLAIR` 双模态脑肿瘤分类客户端完全对应的公开下载源，较可能是原实验私有/半私有医院数据。
- `Yale-Brain-Mets-Longitudinal` 已确认公开存在，包含 T1、T1CE、T2、FLAIR NIfTI，可作为 `brain_metastases` 或外部验证客户端，但体量约 43GB，下载和预处理成本较高。

可替代公开 3D 多模态来源：

- UPENN-GBM: https://www.cancerimagingarchive.net/collection/upenn-gbm/
- UCSF-PDGM: https://www.cancerimagingarchive.net/collection/ucsf-pdgm/
- UTSW-Glioma: https://www.cancerimagingarchive.net/collection/utsw-glioma/
- Pretreat-MetsToBrain-Masks: https://www.cancerimagingarchive.net/collection/pretreat-metstobrain-masks/

## 已提供脚本

### Figshare 原始版

```bash
bash data/scripts/download_figshare.sh
python data/scripts/preprocess_figshare.py \
  --raw_dir data/raw/figshare \
  --output_dir data/processed/Figshare \
  --overwrite
```

说明：原始 Figshare 下载在部分网络环境下可能返回 403。如果遇到这种情况，优先使用下面的 Hugging Face 镜像路径。

### Figshare Hugging Face 镜像版

```bash
python data/scripts/preprocess_figshare_hf.py \
  --output_dir data/processed/Figshare \
  --overwrite
```

该镜像已提供 patient-level 的 train/test 划分，适合快速补充 2D 分类客户端。

### BRISC2025 Zenodo 正式版

下载：

```bash
bash data/scripts/download_brisc2025_zenodo.sh
```

预处理：

```bash
python data/scripts/preprocess_brisc2025.py \
  --zip_path data/raw/brisc2025/brisc2025.zip \
  --output_dir data/processed/Brisc2025 \
  --overwrite
```

该数据包含 `glioma/meningioma/pituitary/no_tumor` 四类，保存为 `t1.npz`，最适合替换当前只用于链路验证的 `Simezu` 部分数据。

### 公开 3D 多模态客户端（已采用，BraTS2023 GLI + MEN）

用 Hugging Face 上 CC BY 4.0 的 BraTS2023 公开 NIfTI 数据构造两个 3D 客户端：

- `BraTS`：四模态全模态 3D 客户端（t1, t1c, t2w, t2f）。
- `Shanghai`：从同源数据抽取 t1c+t2f，模拟双模态 partial-modality 3D 客户端（不是原私有上海医院数据）。

类别：glioma 来自 `Angelou0516/brats2023-gli-dataset`，meningioma 来自 `Angelou0516/brats2023-men-dataset`。

```bash
# 需要 nibabel：python3 -m pip install --target .deps nibabel
PYTHONPATH=.deps python3 data/scripts/preprocess_brats_3d_hf.py \
  --output_root data/processed \
  --brats_cases_per_class 24 \
  --shanghai_cases_per_class 24 \
  --brats_shape 32,112,112 \
  --shanghai_shape 16,112,112 \
  --test_ratio 0.25 \
  --overwrite
```

说明：体数据使用三线性插值重采样到指定分辨率，便于在仅 CPU 的机器上完成训练。论文中需写明这是“用公开多模态数据模拟模态缺失/异构联邦场景”。

### 2D 客户端类平衡子集（CPU 友好）

为让 2D 客户端与 3D 客户端规模可比、且类别均衡，可对 2D 数据做按类平衡下采样并重采样到 128×128：

```bash
python3 data/scripts/preprocess_brisc2025.py \
  --zip_path data/raw/brisc2025/brisc2025.zip \
  --output_dir data/processed/Brisc2025 \
  --image_size 128 --max_per_class_train 300 --max_per_class_test 80 --seed 42 --overwrite

python3 data/scripts/preprocess_figshare_hf.py \
  --output_dir data/processed/Figshare \
  --image_size 128 --max_per_class_train 300 --max_per_class_test 80 --overwrite
```

### 分辨率配置（训练侧）

`Graduation-Design-main/dataset.py` 支持用环境变量覆盖各客户端空间尺寸（默认全分辨率，供 GPU 全量复现）：

```bash
export FDU_BRATS_SHAPE="32,112,112"
export FDU_SHANGHAI_SHAPE="16,112,112"
export FDU_FIGSHARE_SHAPE="128,128"
export FDU_BRISC2025_SHAPE="128,128"
```

实验脚本 `experiments/run_public_4client_*.sh` 已自动设置这些变量。

### Simezu Hugging Face 公开组合数据

限量下载，用于快速构造第二个 2D 客户端：

```bash
python data/scripts/preprocess_simezu_hf.py \
  --output_dir data/processed/Brisc2025 \
  --max_samples_per_class 300
```

全量下载，用于公开可复现实验：

```bash
python data/scripts/preprocess_simezu_hf.py \
  --output_dir data/processed/Brisc2025
```

该脚本把标签映射为 `no_tumor/glioma/meningioma/pituitary`，模态保存为 `t1.npz`。

### 数据统计表

```bash
python data/scripts/summarize_dataset.py \
  --processed_dir data/processed \
  --output_csv paper_outputs/dataset_summary.csv
```

输出可以直接整理成论文中的数据集统计表。

## 后续预处理目标

需要统一完成：

1. 标签映射到项目全局标签：
   - `no_tumor`
   - `meningioma`
   - `glioma`
   - `pituitary`
   - `brain_metastases`
2. 模态命名统一：
   - `t1`
   - `t1c`
   - `t2w`
   - `t2f`
3. 保存为 `.npz`。
4. 固定 train/val/test 划分，避免多次实验数据不一致。
5. 保存数据统计表，包括每客户端样本数、类别数、模态组合和 2D/3D 形态。

## 实验优先级

第一阶段先保证当前 4 客户端复现：

- BraTS
- Shanghai
- Figshare
- Brisc2025

第二阶段加入新设置：

- 不同模态缺失强度。
- 加入 Yale 外部验证。
- 单独构造 2D-only、3D-only、2D+3D 混合场景。
- 按客户端数量扩展为 6、8、10 客户端。
