# 联邦多模态脑肿瘤诊断项目

本项目研究“多中心医学影像在模态缺失、数据异构和模型异构条件下如何进行联邦学习”。简单说，它希望让不同医院在不共享原始 MRI 图像的情况下，共同训练脑肿瘤分类模型。

当前代码主体位于 `Graduation-Design-main/`，毕业论文草稿为 `基于联邦学习的模态缺失异构脑肿瘤诊断[副本].docx`，相关论文 PDF 已整理到 `文献资料/`。

## 项目目标

真实医院的数据通常不整齐：

- 有的医院有完整 3D MRI：T1、T1c、T2、FLAIR。
- 有的医院只有部分模态，例如 T1c + FLAIR。
- 有的公开数据集只有 2D 单模态图像。
- 各医院类别比例、样本数量和图像形态也不同。

本项目提出 `FedMFG`，目标是在这种不整齐的数据环境中完成联邦脑肿瘤分类。

## 核心创新点

`FedMFG` 的全称是 `Federated Modality-aware Fusion and prototype-Guided head aggregation`。

它的主要思想有三点：

1. **模态感知融合**：客户端只使用真实存在的 MRI 模态，不强行补零，也不生成虚假的缺失模态。
2. **模态组合级类别原型**：服务器不只保存“某一类别”的原型，而是保存“某一模态组合 + 某一类别”的原型，例如 `(T1c+FLAIR, glioma)`。
3. **原型指导分类头聚合**：不直接平均所有客户端的完整模型，只聚合统一维度的分类头，并根据本地原型与全局原型的一致性决定每个客户端更新的可信度。

这个设计适合处理 2D/3D 数据并存、模态组合不同、客户端模型结构不完全一致的场景。

## 代码结构

```text
Graduation-Design-main/
  train.py                 # 训练入口
  test.py                  # 测试入口
  dataset.py               # 数据集读取、模态定义、标签定义
  model.py                 # ResNet2D/3D、MMModel、AMMModel、MFGModel
  loss.py                  # 分类损失和原型损失
  client/                  # 各算法的客户端训练逻辑
  server/                  # 各算法的服务器聚合逻辑
  scripts/train/           # 训练脚本
  scripts/test/            # 测试脚本
  *_history.json           # 已有实验记录
```

## 支持的数据客户端

代码中默认使用 4 个客户端：

| 客户端 | 模态 | 数据形态 |
| --- | --- | --- |
| BraTS | T1, T1c, T2, FLAIR | 3D |
| Shanghai | T1c, FLAIR | 3D |
| Figshare | T1c | 2D |
| Brisc2025 | T1 | 2D |

代码中还预留了 `Yale` 客户端，可通过 `--include_yale` 启用。

## 已实现算法

训练入口 `train.py` 支持：

- `fedmfg`：本项目主方法。
- `fedamm`：任意模态缺失相关对比方法。
- `fedmm`：模态缺失联邦对比方法。
- `fedproto`：原型联邦学习。
- `fedtgp`：可训练全局原型方法。
- `fedgh`：全局分类头方法。
- `fd`：特征蒸馏类方法。
- `local`：各客户端本地独立训练。

## 运行环境

建议环境：

- Python 3.10 或更高版本
- PyTorch
- MONAI
- NumPy
- scikit-learn
- matplotlib
- tqdm

安装示例：

```bash
pip install torch monai numpy scikit-learn matplotlib tqdm
```

如果使用 GPU，请根据 CUDA 版本安装对应的 PyTorch。

## 数据目录要求

默认数据根目录已经改为项目内相对路径：

```text
data/processed
```

数据组织方式应类似：

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

每个 `.npz` 文件需要包含键 `x`，表示已经预处理好的图像数组。

## 训练主方法

运行 `FedMFG`：

```bash
cd Graduation-Design-main
bash scripts/train/train_mfg.sh
```

如果数据路径不同，可以覆盖参数：

```bash
bash scripts/train/train_mfg.sh --root_dir /你的数据路径
```

测试：

```bash
cd Graduation-Design-main
bash scripts/test/test_mfg.sh --checkpoint checkpoints/fedmfg/best_checkpoint.pth
```

## 当前实验结果

毕业论文草稿中记录的主实验结果如下：

| 算法 | Accuracy | Macro F1 |
| --- | ---: | ---: |
| Solo | 84.25 | 84.05 |
| FD | 80.79 | 76.62 |
| FedGH | 73.18 | 68.28 |
| FedProto | 83.95 | 75.93 |
| FedTGP | 87.63 | 87.01 |
| FedMM | 78.58 | 77.50 |
| FedAMM | 85.91 | 85.27 |
| FedMFG | 90.25 | 90.29 |

当前 JSON 结果中，`fd`、`fedgh`、`fedmm`、`fedproto`、`fedtgp` 和 `local` 已有测试记录；`fedmfg_history.json` 主要记录了验证集结果，尚未完整保存最终测试结果。因此开源和投稿前，需要重新跑一次标准测试并导出正式结果 JSON。

## 当前推进状态

已完成：

- 建立 `data/` 数据目录规范。
- 完成 `Figshare` Hugging Face 镜像预处理，得到 `data/processed/Figshare`。
- 新增 `Simezu/brain-tumour-MRI-scan` 预处理脚本，可作为 `Brisc2025` 的公开替代路线。
- 完成正式 `BRISC2025` Zenodo 数据下载与预处理，得到 `data/processed/Brisc2025`，共 train=5000、test=1000。
- 跑通 `Figshare` 单客户端 FedMFG smoke test。
- 跑通 `Figshare + Brisc2025替代` 双客户端 FedMFG smoke test。
- 跑通 `Figshare + 正式 BRISC2025` 公开双客户端 baseline 链路检查，输出 `paper_outputs/public_2client/summary.csv`。
- 新增论文初稿框架：`paper/outline.md`。
- 新增实验追踪表：`paper/experiment_tracker.md`。

仍然阻塞正式论文实验的关键问题：

- 当前机器仍然没有 `BraTS` 和 `Shanghai` 原始/预处理数据。
- 当前机器无 CUDA/MPS，只能 CPU 训练；全量正式实验会很慢，建议使用 GPU 或分批长时间运行。
- 现有 smoke test 和 `MAX_SAMPLES=80` 链路检查只用于验证代码链路，不可作为论文正式结果。

## 论文潜力判断

这个项目具备形成论文的基础，但目前更接近“毕业论文/初稿工程”阶段。要投会议或期刊，还需要补强：

- 固定随机种子，多次重复实验，报告均值和标准差。
- 补齐 `FedMFG` 的正式测试 JSON、混淆矩阵和每客户端结果。
- 与最近的 `FedAMM`、`FedMEMA`、`PmcmFL`、`MFCPL` 等论文做更严格对比。
- 增加消融实验的代码入口，而不是只在论文中写结果。
- 明确公开数据、私有数据、预处理流程和伦理合规说明。
- 修复工程问题，例如无效导入、训练脚本中硬编码 checkpoint、`train_amm.sh` 首行异常字符。

更详细的投稿分析见 `论文推进/论文潜力与实验补强计划.md`。

## GitHub 维护建议

建议上传到一个新的 GitHub 仓库，例如：

```text
Eternity1212/fl-agent
```

或更清晰的名称：

```text
Eternity1212/FedMFG-BrainTumor
```

后续每完成一次代码修复、实验补充或结果更新，都应该：

```bash
git status
git add .
git commit -m "说明本次修改"
git push
```

我可以继续负责把项目初始化为 Git 仓库、创建远程仓库、首次推送，并在后续修改后同步推送。
