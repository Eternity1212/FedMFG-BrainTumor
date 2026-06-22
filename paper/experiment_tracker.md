# 实验追踪表

更新时间：2026-06-22（晚间更新：补齐公开 3D 客户端 + 低分辨率 CPU 可跑数据）

## 数据状态

| 数据/客户端 | 状态 | 本地路径 | 说明 |
| --- | --- | --- | --- |
| Figshare | 已完成（128×128） | `data/processed/Figshare` | Hugging Face 镜像，2D t1c，train=2522, test=542，3 类 |
| BraTS | 已用公开数据替代完成 | `data/processed/BraTS` | BraTS2023 GLI(glioma)+MEN(meningioma)，3D 四模态 32×112×112，train=36, test=12，2 类 |
| Shanghai | 已用公开数据替代完成 | `data/processed/Shanghai` | 同源 BraTS2023 抽取 t1c+t2f 模拟双模态 3D 客户端 16×112×112，train=36, test=12，2 类 |
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
| seed42 baseline（local/fedgh/fedproto/fedtgp/fedmm/fedamm/fedmfg） | 进行中 | `paper_outputs/public_4client/summary_seed42.csv` |
| seed43 baseline | 待运行 | `paper_outputs/public_4client/summary_seed43.csv` |
| seed44 baseline | 待运行 | `paper_outputs/public_4client/summary_seed44.csv` |
| 多 seed 汇总 | 待运行 | `paper_outputs/public_4client/summary_all_seeds.csv` |
| FedMFG 消融 seed42 | 待运行 | `paper_outputs/public_4client_ablation/summary_seed42.csv` |

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
