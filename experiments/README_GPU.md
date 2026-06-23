# 远程 GPU 全分辨率部署指南（对标原文 90%）

CPU 机器上的实验是“能跑通 + 验证方法改进方向”的缩水版（低分辨率、少样本、10 轮、lr 3e-4）。
要真正对标原论文的 90.25% / Macro-F1 90.29%，必须在 GPU 上用**全分辨率 + 全量数据 + 16 轮 + lr 1e-3**。
本指南给出在一台远程 GPU 服务器（理想 A100，单卡亦可）上的完整步骤。

## 0. 原文实验配置（目标对标）
- 硬件：8× A100 80GB（单卡也能跑，调小 batch 即可）
- 骨干：ResNet-18（MONAI），优化器 AdamW，学习率 **1e-3**，**16 轮**
- 分辨率：BraTS 3D **155×224×224**、Shanghai 3D 全网格、2D **512×512**
- 4 客户端：BraTS(全四模态3D) / Shanghai(双模态3D) / Figshare(2D,3类) / Brisc2025(2D,4类)
- 评测主指标：**客户端宏平均 Accuracy 与 Macro-F1**（不依赖大样本客户端）

## 1. 拉代码 + 装环境
```bash
git clone https://github.com/Eternity1212/FedMFG-BrainTumor.git
cd FedMFG-BrainTumor

python3 -m venv .venv && source .venv/bin/activate
# 按服务器 CUDA 版本安装对应 torch，例如 CUDA 12.1：
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r Graduation-Design-main/requirements.txt
# 3D NIfTI 预处理需要 nibabel
pip install nibabel huggingface_hub
```
验证 GPU：`python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`

## 2. 准备全分辨率数据
2D（Figshare / Brisc2025）保持 512×512（不要传 `--image_size`，或显式设 512）：
```bash
python3 data/scripts/preprocess_figshare_hf.py            # 全部 3064 张，原尺寸
python3 data/scripts/preprocess_brisc2025.py --image_size 512   # 6000 张
```
3D（BraTS / Shanghai）用**全分辨率**、更多 case（不要降到 32×112×112）：
```bash
python3 data/scripts/preprocess_brats_3d_hf.py \
  --brats_cases_per_class 200 --shanghai_cases_per_class 200 \
  --brats_shape 155,224,224 --shanghai_shape 155,224,224 \
  --test_ratio 0.2
```
> 若有真实 BraTS 全量 / 真实 Shanghai 私有数据，按 `data/README.md` 的目录结构放入
> `data/processed/{BraTS,Shanghai}/{train,test}/<label>/<case>/<modality>.npz` 即可，效果更接近原文。

## 3. 一键跑全分辨率多 seed 套件
```bash
SEEDS="42 43 44" ROUNDS=16 DEVICE=cuda \
  bash experiments/run_gpu_fullres_suite.sh
```
- 该脚本**不**设置 `FDU_*_SHAPE`，因此 `dataset.py` 使用全分辨率默认值。
- 自动开启 AMP、AdamW lr=1e-3、16 轮，并对 fedgh/fedproto/fedtgp 用 `--model_mode auto`（单编码器，避免异构维度崩溃），fedmfg 用改进后的 `count_rho_eta` 头部聚合。
- 显存不足就调小：`CBS_MAP="BraTS=2 Shanghai=4 Figshare=32 Brisc2025=32"`。

## 4. 取结果
- 每 seed：`paper_outputs/gpu_fullres/summary_seed*.csv`
- 多 seed 汇总（含均值±std、逐客户端）：`paper_outputs/gpu_fullres/summary_all_seeds.csv`
- 客户端宏平均 / Macro-F1 用 `paper_tools/aggregate_multiseed.py` 已输出；论文主表以此为准。

## 5. 消融（可选）
```bash
SEED=42 ROUNDS=16 bash experiments/run_public_4client_mfg_ablation.sh
# 注意：消融脚本默认带 FDU_*_SHAPE 低分辨率，GPU 全分辨率时请先 unset 这些环境变量
```

## 6. 把结果同步回来
```bash
git add paper_outputs/gpu_fullres/summary_*.csv paper/results/ paper/experiment_tracker.md
git commit -m "results(gpu): full-resolution multi-seed main table"
git push
```
（大文件如 checkpoints/plots 已被 .gitignore 忽略，只提交汇总 CSV。）
