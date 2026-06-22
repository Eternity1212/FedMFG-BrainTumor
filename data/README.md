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

### 4. Shanghai / Yale 私有或半公开数据

用途：作为双模态或外部验证客户端。

注意：如果涉及私有医学数据，论文中必须说明伦理审批、脱敏方式、数据不可公开原因。

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
