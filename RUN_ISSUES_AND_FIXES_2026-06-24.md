# 2026-06-24 运行问题与处理记录

本文记录本次在 `FedMFG-BrainTumor` 仓库上从数据准备到运行实验过程中遇到的问题、处理过程、当前阻塞点，以及对仓库源码所做的修改。

## 1. 本次遇到的主要问题

### 1.1 历史首次失败：外网数据下载阶段 SSL/HTTPS 异常
用户最初提供的报错显示，以下数据源在下载或列目录时失败：

- Hugging Face：
  - `Angelou0516/Figshare_Brain_Tumor`
  - `Angelou0516/brats2023-gli-dataset`
- Zenodo：
  - `BRISC2025`

典型报错包括：

- `SSLError(SSLEOFError(... EOF occurred in violation of protocol ...))`
- `ConnectionError: Couldn't reach 'Angelou0516/Figshare_Brain_Tumor' on the Hub (SSLError)`
- `urllib.error.URLError: <urlopen error EOF occurred in violation of protocol (_ssl.c:1123)>`

这些错误导致：

- `Figshare` 未成功预处理
- `Brisc2025` 的 `brisc2025.zip` 缺失
- `BraTS / Shanghai` 3D 公共数据未成功预处理
- 随后的训练阶段找不到 `data/processed/BraTS/train`

### 1.2 本次实际接手后，数据下载已恢复正常
在本次会话中重新检查后：

- `https://huggingface.co` 可访问
- `https://zenodo.org` 可访问
- `datasets.load_dataset('Angelou0516/Figshare_Brain_Tumor')` 可以成功加载
- `BRISC2025` 可以成功重新下载

说明：

- 初始失败更像是一次网络/SSL 环境问题，而不是当前仓库脚本永久失效

### 1.3 串行 3D 预处理导致 Shanghai 等待 BraTS 完成
原始 `data/scripts/preprocess_brats_3d_hf.py` 会在一个脚本中顺序处理：

1. `BraTS`
2. `Shanghai`

这意味着即使 `BraTS` 较慢，也必须等它跑到后半段后，`Shanghai` 才会开始生成。

### 1.4 数据全部补齐后，训练阶段新的阻塞点是 GPU OOM
在数据阶段全部完成后，重新启动训练：

```bash
PYTHON=python3 DO_ENV=0 DO_DATA=0 DO_TRAIN=1 DO_ABLATION=1 DO_REPORT=1 bash run.sh
```

训练已成功进入 `STAGE 3`，但在 `seed=42` 的多个算法上，第一轮 BraTS 训练很快就出现：

- `torch.OutOfMemoryError: CUDA out of memory`

典型报错信息显示：

- GPU 总显存：`79.11 GiB`
- 可用显存只剩约：`2.08 GiB`
- 同卡上已有其他进程占用大量显存：
  - 一个约 `63.21 GiB`
  - 一个约 `12.60 GiB`
  - 一个约 `1.19 GiB`

因此当前失败主因不是数据缺失，而是：

- **运行训练时 GPU 上已有其他大任务占用显存，导致本实验在 BraTS 第一个 batch 反向传播时 OOM。**

## 2. 本次处理方法

### 2.1 重新执行数据阶段，只跳过环境安装
为了避免在数据不完整时直接重试训练，先单独执行数据阶段：

```bash
PYTHON=python3 DO_ENV=0 DO_DATA=1 DO_TRAIN=0 DO_ABLATION=0 DO_REPORT=0 bash run.sh
```

处理结果：

- `Figshare` 成功完成
- `Brisc2025` 成功下载并完成预处理
- `BraTS / Shanghai` 进入 3D 全分辨率预处理

### 2.2 手动重新下载 BRISC2025 原始压缩包
单独执行：

```bash
bash data/scripts/download_brisc2025_zenodo.sh
```

结果：

- `data/raw/brisc2025/brisc2025.zip` 成功下载

### 2.3 为了加快 3D 数据准备，新增 Shanghai-only 并行预处理能力
为避免 `Shanghai` 一直等待 `BraTS` 串行完成，对 3D 预处理脚本做了最小修改，使其支持只处理某一个客户端。

新增能力后，可执行：

```bash
python3 data/scripts/preprocess_brats_3d_hf.py \
  --output_root data/processed \
  --metadata_csv paper_outputs/shanghai_public_cases.csv \
  --clients Shanghai \
  --brats_cases_per_class 150 \
  --shanghai_cases_per_class 150 \
  --brats_shape 155,224,224 \
  --shanghai_shape 155,224,224 \
  --test_ratio 0.2 \
  --overwrite
```

这使得：

- 原始 `run.sh` 的数据阶段继续按原逻辑推进 `BraTS`
- 新开一个后台进程并行生成 `Shanghai`
- 两者分别写入不同输出目录：
  - `data/processed/BraTS`
  - `data/processed/Shanghai`

### 2.4 最终数据阶段完成情况
最终四个客户端数据全部补齐：

- `Figshare`：完整
- `Brisc2025`：完整
- `BraTS`：
  - train `240`
  - test `60`
- `Shanghai`：
  - train `240`
  - test `60`

并成功生成：

- `paper_outputs/gpu_fullres/dataset_summary.csv`

## 3. 当前实验运行状态

### 3.1 数据阶段
已完成。

### 3.2 主实验 / 消融 / 汇总阶段
已启动，但目前被 GPU 显存不足阻塞。

当前完整运行命令为：

```bash
PYTHON=python3 DO_ENV=0 DO_DATA=0 DO_TRAIN=1 DO_ABLATION=1 DO_REPORT=1 bash run.sh
```

### 3.3 当前可见结果
截至本记录写入时：

- 训练已进入 `STAGE 3`
- `seed=42` 的多个算法（如 `fedmfg`, `fedamm`, `fedmm`）已因 OOM 失败
- `paper_outputs/gpu_fullres/histories/` 仍未生成有效 `*_history.json`
- `paper/results/gpu_main_report.csv` 与 `paper/results/gpu_ablation_report.csv` 当前仅为空壳/表头文件

## 4. 对源码是否做过修改

### 4.1 做过的源码修改
本次会话中，确实对仓库源码做了 **1 处功能性修改**：

- 修改文件：`data/scripts/preprocess_brats_3d_hf.py`

修改目的：

- 为了支持 `BraTS` / `Shanghai` 分开预处理
- 让 `Shanghai` 可以在不干扰原始 `run.sh` 数据进程的前提下并行生成

修改内容概述：

- 新增参数：`--clients`
- 新增客户端解析逻辑：`parse_clients(...)`
- 将原先固定顺序的双客户端处理改为“按指定客户端列表处理”

### 4.2 不属于源码修改的内容
以下内容是下载或运行产物，不属于源码逻辑变更：

- `data/raw/brisc2025/brisc2025.zip`
- `paper_outputs/gpu_fullres/dataset_summary.csv`
- `paper/results/gpu_main_report.csv`
- `paper/results/gpu_ablation_report.csv`
- `paper_outputs/gpu_fullres/` 下的中间输出目录

## 5. 建议的后续处理方式

### 5.1 若继续使用当前全分辨率配置运行
需要先处理 GPU 资源问题，建议优先执行：

```bash
nvidia-smi
```

确认并释放当前占用大显存的其他进程后，再重新运行训练。

### 5.2 若无法清理当前 GPU 占用
则需要降低当前实验配置，例如：

- 调小 `CBS_MAP`
  - 例如：`BraTS=2 Shanghai=4 Figshare=32 Brisc2025=32`
- 或进一步降低 3D 分辨率 / case 数

但这会影响与“全分辨率对标原文”的一致性。

## 6. 一句话总结

本次问题演变过程如下：

1. 最初失败是 **外网 SSL / HTTPS 异常** 导致数据不完整；
2. 本次会话中已成功补齐全部数据；
3. 为提高效率，新增了 `Shanghai` 的并行 3D 预处理能力；
4. 当前新的阻塞点已经变成 **GPU 显存被其他进程占满，导致训练 OOM**。
