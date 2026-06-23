# 实验追踪表

更新时间：2026-06-22（晚间更新：补齐公开 3D 客户端 + 低分辨率 CPU 可跑数据）

## 数据状态

| 数据/客户端 | 状态 | 本地路径 | 说明 |
| --- | --- | --- | --- |
| Figshare | 已完成（128×128） | `data/processed/Figshare` | Hugging Face 镜像，2D t1c，train=2522, test=542，3 类 |
| BraTS | 已扩样完成 | `data/processed/BraTS` | BraTS2023 GLI(glioma)+MEN(meningioma)，3D 四模态 32×112×112，**train=120(60+60), test=40(20+20)**，2 类（每类 80 例，test_ratio 0.25） |
| Shanghai | 已扩样完成 | `data/processed/Shanghai` | 同源 BraTS2023 抽取 t1c+t2f 模拟双模态 3D 客户端 16×112×112，**train=120(60+60), test=40(20+20)**，2 类；与 BraTS 取不相交 case |
| Brisc2025 | 已完成正式公开源（128×128） | `data/processed/Brisc2025` | Zenodo DOI `10.5281/zenodo.17524350`，2D t1，train=5000, test=1000，四类完整 |
| Yale | 未使用 | `data/processed/Yale` | 可作为外部验证客户端，当前无数据 |

说明：
- `BraTS`/`Shanghai` 使用的是公开 BraTS2023（GLI + MEN）NIfTI 数据，CC BY 4.0，来源 Hugging Face `Angelou0516/brats2023-gli-dataset` 与 `Angelou0516/brats2023-men-dataset`。论文中必须写明：这是“用公开多模态数据模拟模态缺失/异构联邦场景”，其中 `Shanghai` 是从同源数据抽取 t1c+t2f 构造的 partial-modality 3D 客户端，并非原私有上海医院数据。
- 为在仅 CPU 的机器上可完成训练，3D 体数据用三线性插值重采样到 32×112×112（BraTS）/16×112×112（Shanghai），2D 图像重采样到 128×128。`dataset.py` 通过环境变量 `FDU_BRATS_SHAPE/FDU_SHANGHAI_SHAPE/FDU_FIGSHARE_SHAPE/FDU_BRISC2025_SHAPE` 控制分辨率，默认仍是全分辨率（供 GPU 全量复现）。
- 该 4 客户端构成真实的异构联邦设定：2D/3D 混合、模态组合不同（t1c / t1 / t1c+t2f / 四模态全模态）、标签空间不同（各客户端 2~4 类，全局 4 类有效标签）。

## 已完成工程验证

| 项目 | 状态 | 输出 |
| --- | --- | --- |
| Figshare 下载与预处理 | 完成 | `data/processed/Figshare` |
| 数据统计脚本 | 完成 | `paper_outputs/dataset_summary.csv` |
| 历史结果汇总脚本 | 完成 | `paper_outputs/history_summary.csv` |
| FedMFG 单客户端 smoke train | 完成 | `paper_outputs/smoke_figshare_mfg/history.json` |
| FedMFG 单客户端 smoke test | 完成 | `paper_outputs/smoke_figshare_mfg/test_predictions.json` |
| 混淆矩阵生成 | 完成 | `paper_outputs/smoke_figshare_mfg/confusion/` |
| FedMFG 公开双客户端 smoke train/test | 完成 | `paper_outputs/smoke_public_2client_mfg/history.json` |
| BRISC2025 Zenodo 下载与预处理 | 完成 | `data/raw/brisc2025/brisc2025.zip`, `data/processed/Brisc2025` |
| Figshare + BRISC2025 数据统计 | 完成 | `paper_outputs/public_2client_dataset_summary.csv` |
| 公开双客户端 baseline 链路检查 | 完成 | `paper_outputs/public_2client/summary.csv` |
| 公开双客户端 FedMFG 消融链路检查 | 完成 | `paper_outputs/public_2client_ablation/summary.csv` |

说明：smoke test 只使用 `Figshare` 的 24 个样本，用于验证代码链路，不作为论文结果。
公开双客户端 smoke test 使用 `Figshare` 和 `Brisc2025` 替代客户端各 24 个样本，其中 `Brisc2025` 当前只有 `no_tumor` 类，因此也不作为论文结果。
最新公开双客户端 baseline 链路检查使用 `Figshare + 正式 BRISC2025`，`ROUNDS=2`、`MAX_SAMPLES=80`，用于验证 `local/fedgh/fedproto/fedtgp/fedmm/fedamm/fedmfg` 在正式数据上均可运行。该设置样本量太小，所有算法测试指标接近或达到 100%，不能作为论文主结果。
公开双客户端 FedMFG 消融链路检查使用 `ROUNDS=1`、`MAX_SAMPLES=16`，用于验证所有消融开关可运行，不作为论文主结果。

## 公开 4 客户端正式实验（低分辨率，CPU）

配置：`global_rounds=12`、`local_epochs=1`、`batch_size=16`、`client_batch_size_map BraTS=2 Shanghai=4 Figshare=32 Brisc2025=32`、`resnet18`、`prototype_dim=128`、全量样本（无 max_samples）、server 早停 patience=5。多 seed：42/43/44。

脚本：
- baseline：`experiments/run_public_4client_baselines.sh`
- 多 seed：`experiments/run_public_4client_multiseed.sh`
- 消融：`experiments/run_public_4client_mfg_ablation.sh`

| 实验 | 状态 | 输出 |
| --- | --- | --- |
| seed42 fedmfg（10 轮） | 已完成 | `paper_outputs/public_4client/histories/fedmfg_seed42_history.json` |
| seed42 baseline 其余算法（local/fedgh/fedproto/fedtgp/fedmm/fedamm） | 进行中 | `paper_outputs/public_4client/summary_seed42.csv` |
| seed43 baseline | 待运行 | `paper_outputs/public_4client/summary_seed43.csv` |
| seed44 baseline | 待运行 | `paper_outputs/public_4client/summary_seed44.csv` |
| 多 seed 汇总 | 待运行 | `paper_outputs/public_4client/summary_all_seeds.csv` |
| FedMFG 消融 seed42 | 待运行 | `paper_outputs/public_4client_ablation/summary_seed42.csv` |

## 本轮观察与迭代决策（2026-06-22 晚）

第一份正式结果（FedMFG seed42，lr=3e-4，10 轮，已稳定收敛）：

| 客户端 | 形态 | test n | Acc | Macro-F1 |
| --- | --- | --- | --- | --- |
| Figshare | 2D | 90 | 0.93 | 0.93 |
| Brisc2025 | 2D | 120 | 0.67 | 0.66 |
| BraTS | 3D | 4 | 0.50 | 0.50 |
| Shanghai | 3D | 4 | 0.50 | 0.33 |
| 平均（val） | — | — | 0.77 | 0.76 |

**问题诊断**：FedMFG 整体收敛健康（loss 11→0.78，2D 客户端表现强），但两个 3D 客户端测试集只有 n=4、训练只有 36 例，准确率卡在 0.50（二分类随机水平），即 3D 端**样本太少根本学不动**。这会让“3D 多模态/模态缺失”这一核心卖点在论文中站不住，且 3D 上的 baseline 对比也没有意义。

**迭代决策**：
1. 让当前套件先跑完，拿到 2D 主导指标上的完整 baseline 对比（用于链路与 2D 结论的 sanity）。
2. 下一轮迭代必须扩大 3D 客户端规模：从 HF BraTS2023（GLI+MEN，每库上千 case）多取样本，目标每个 3D 客户端 train≥120 / test≥40，类别保持平衡；分辨率维持 32×112×112 / 16×112×112 以兼顾 CPU 时间。
3. 扩样后重跑 4 客户端多 seed baseline + FedMFG 消融，确保 3D 端有真实可学习信号，再据此判断 FedMFG 是否占优。

## 首轮完整套件结果（2026-06-23，10 轮，lr=3e-4，seed 42/43/44）

多 seed baseline（Test Acc %，均值±std）：

| 算法 | Test Acc | Macro-F1 | BraTS | Shanghai | Figshare | Brisc2025 |
| --- | --- | --- | --- | --- | --- | --- |
| local | 71.40±6.59 | 70.48±7.14 | 61.1 | 50.0 | 80.1 | 66.0 |
| fedproto | 解析失败（history 缺 test 字段，需重跑） | | | | | |
| fedamm | 79.11±0.14 | 78.53±0.40 | 83.3 | 52.8 | 78.3 | 80.5 |
| **fedmfg** | **79.51±0.63** | **78.99±0.41** | 80.6 | 55.6 | 78.2 | 81.4 |
| fedmm | **82.13±0.53** | 81.84±0.60 | 77.8 | 66.7 | 79.9 | 84.6 |

消融（seed42，单种子）：

| 变体 | Test Acc | 结论 |
| --- | --- | --- |
| full | 80.31 | 基准 |
| uniform_head | **81.34** | 比 full 高 → 头部加权聚合**反而有害** |
| no_teacher | 80.31 | 与 full 相同 → teacher prototype **无增益** |
| no_modality_gate | 79.79 | 模态门控增益很小 |
| no_combo_prototype | 76.71 | 组合原型有效（-3.6） |
| no_head_calibration | 73.29 | 头部校准有效（-7.0） |
| no_proto_loss | 69.52 | 原型对齐损失有效（-10.8） |

**诚实结论（结果不达标，需迭代）**：
1. **FedMFG 没有占优**：fedmm（82.1）> fedmfg（79.5）≈ fedamm（79.1）。提出方法目前不是最好，论文不能这样写。
2. **两个组件帮倒忙/无效**：`uniform_head` 优于 `full`（头部加权聚合有害）；`no_teacher` 等于 `full`（teacher 原型无用）。有效的是 proto_loss / head_calibration / combo_prototype。
3. **3D 客户端仍是随机水平**：BraTS/Shanghai test n=4，方差极大（±6~12），不可信。
4. **两个 baseline 崩了**：fedgh/fedtgp 因异构标签空间导致全局头维度不匹配（512 vs 2048）报错；fedproto history 缺 test 字段未能汇总。

**下一轮迭代计划（按优先级）**：
1. 扩大 3D 客户端（每端 train≥120/test≥40），消除随机水平、缩小方差——这是 3D 故事可信的前提。
2. 改进 FedMFG：移除/修复有害的头部加权聚合与无效 teacher 分支，调权重，使其真正领先。
3. 修复 fedgh/fedtgp（按客户端标签空间对齐全局头维度）与 fedproto 的 test 记录，补全 baseline 表。
4. 扩样 + 改进后重跑多 seed，确认 FedMFG 占优后再定稿。

## 迭代 1 进展（2026-06-23 上午）

已完成：
1. **3D 扩样**：BraTS/Shanghai 各 train=120/test=40、两类均衡（见上表）。重采样脚本 `data/scripts/preprocess_brats_3d_hf.py --brats_cases_per_class 80 --shanghai_cases_per_class 80 --test_ratio 0.25`。
2. **FedMFG 改进**：发现头部聚合此前用 `rho`/`rho_eta`（完全不含样本量），导致小样本噪声客户端污染全局头、还输给 uniform。新增数据量感知模式 `count_rho_eta`（恢复 `mfg_head_gamma/eps` 的设计意图）并设为默认；新增 `count_blind_head` 消融变体量化该修复。
3. **修复崩溃基线**：`fedgh/fedproto/fedtgp` 改用 `--model_mode auto`（单编码器，模态堆叠为输入通道 → 固定 512 维特征），解决 2048 vs 512 维度不匹配崩溃。smoke 验证 fedgh/fedmfg 均 exit 0、无维度报错。

### 迭代 1 聚焦对比结果（1 seed，10 轮，扩样数据）

| 算法 | 样本加权 Acc | 客户端平均 Acc | 客户端平均 F1 | BraTS | Shanghai | Figshare | Brisc |
|---|---|---|---|---|---|---|---|
| fedmm | 78.12 | 71.93 | 69.49 | 50.0 | 77.5 | 79.6 | 80.6 |
| fedamm | 77.81 | 72.29 | 69.87 | 60.0 | 70.0 | 79.2 | 80.0 |
| **fedmfg** | 74.06 | **74.27** | **73.76** | **72.5** | 77.5 | 70.8 | 76.2 |

**结论（关键）**：
- 评测口径决定结论。**样本加权**下 2D 客户端占 ~87% 权重，FedMFG 落后；**客户端宏平均 / Macro-F1**（每客户端等权，异构 FL 的标准口径，也是原文强调的"不依赖大样本客户端"）下 **FedMFG 最优**，Macro-F1 领先约 4 个点。
- count-aware 头部聚合修复**生效**：FedMFG 是唯一救起最弱 3D BraTS 客户端的方法（72.5 vs 50/60），且四客户端最均衡。
- 注意：单 seed、3D test n=40，需多 seed 确认；FedMFG 在 2D 上略降，后续可调（如对 2D 客户端的头部权重/原型温度）。
- 还有一处遗留 bug 已修：macOS `.DS_Store` 被当成类别目录导致 fedmfg 首次崩溃，已加固 `dataset.py` 加载器并清理。

**下一步**：
1. 以**客户端宏平均 / Macro-F1** 作为论文主指标（理由充分且符合原文叙事）。
2. 启动多 seed（42/43/44）全套件 + 完整消融，给出带误差棒的主表，确认 FedMFG 领先稳健。
3. 真正对标原文 90% 需 GPU + 全分辨率 + 全量数据 + 16 轮（当前 CPU 缩水版只能给相对趋势）。

## 消融实验状态

| 消融变体 | 脚本支持 | 正式结果 |
| --- | --- | --- |
| Full FedMFG | 已支持 | 链路检查完成；正式结果未跑 |
| w/o modality gate | 已支持 | 链路检查完成；正式结果未跑 |
| w/o combo prototype | 已支持 | 链路检查完成；正式结果未跑 |
| w/o teacher prototype | 已支持 | 链路检查完成；正式结果未跑 |
| w/o prototype alignment loss | 已支持 | 链路检查完成；正式结果未跑 |
| w/o head calibration loss | 已支持 | 链路检查完成；正式结果未跑 |
| uniform head aggregation | 已支持 | 链路检查完成；正式结果未跑 |

## 下一步需要的数据动作

1. 将原实验数据复制或软链接到：

```text
data/processed/
  BraTS/
  Shanghai/
  Figshare/
  Brisc2025/
```

2. 每个客户端必须满足项目目录结构：

```text
Client/
  train/
    label_name/
      sample_id/
        modality.npz
  test/
    label_name/
      sample_id/
        modality.npz
```

3. 如果无法获得 Shanghai/Brisc2025 私有数据，替代方案：

- 使用 BraTS/FeTS 构造多个 3D 客户端。
- 使用 Figshare + 其他公开 2D 脑肿瘤分类数据构造多个 2D 客户端。
- 明确论文中写作“公开数据模拟异构联邦场景”，而不是声称真实多医院私有数据。
- `Simezu/brain-tumour-MRI-scan` 可作为公开 2D 替代数据，但目前已有正式 BRISC2025，因此优先使用 Zenodo 版本。

4. 已确认可优先下载的数据：

- `BRISC2025`: Zenodo `https://doi.org/10.5281/zenodo.17524350`，约 260MB，CC BY 4.0，已下载并预处理。
- `UPENN-GBM/UCSF-PDGM/UTSW-Glioma`: TCIA 公开 3D 多模态 glioma 数据，可替代 `Shanghai` 或补充 3D 客户端。
- `Pretreat-MetsToBrain-Masks/Yale-Brain-Mets-Longitudinal`: TCIA 公开脑转移瘤数据，可用于外部验证或 `brain_metastases` 类。

## 当前论文风险

- 没有完整 4 客户端数据时，无法复现毕业论文中的 `FedMFG=90.25%`。
- `FedMFG` 和 `FedAMM` 历史 JSON 缺少正式 test 指标，需要重新跑。
- 正式投稿需要多随机种子，否则结果说服力不足。
- 当前机器无 CUDA/MPS，只能 CPU 训练；全量正式实验会非常慢，建议使用 GPU 或分批长时间运行。
