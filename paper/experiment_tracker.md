# 实验追踪表

更新时间：2026-06-22

## 数据状态

| 数据/客户端 | 状态 | 本地路径 | 说明 |
| --- | --- | --- | --- |
| Figshare | 已完成 | `data/processed/Figshare` | Hugging Face 镜像，train=2522, test=542 |
| BraTS | 未找到 | `data/processed/BraTS` | 当前机器没有原实验数据；需要官方授权或提供已预处理数据 |
| Shanghai | 未找到 | `data/processed/Shanghai` | 当前机器没有原实验数据；可能是私有数据 |
| Brisc2025 | 未找到 | `data/processed/Brisc2025` | 当前机器没有原实验数据；需要确认来源和许可证 |
| Yale | 未找到 | `data/processed/Yale` | 可作为外部验证客户端，当前无数据 |

## 已完成工程验证

| 项目 | 状态 | 输出 |
| --- | --- | --- |
| Figshare 下载与预处理 | 完成 | `data/processed/Figshare` |
| 数据统计脚本 | 完成 | `paper_outputs/dataset_summary.csv` |
| 历史结果汇总脚本 | 完成 | `paper_outputs/history_summary.csv` |
| FedMFG 单客户端 smoke train | 完成 | `paper_outputs/smoke_figshare_mfg/history.json` |
| FedMFG 单客户端 smoke test | 完成 | `paper_outputs/smoke_figshare_mfg/test_predictions.json` |
| 混淆矩阵生成 | 完成 | `paper_outputs/smoke_figshare_mfg/confusion/` |

说明：smoke test 只使用 `Figshare` 的 24 个样本，用于验证代码链路，不作为论文结果。

## 主实验状态

| 实验 | 状态 | 阻塞原因 |
| --- | --- | --- |
| Local 4 客户端正式复现 | 未开始 | 缺少 BraTS/Shanghai/Brisc2025 |
| FD 4 客户端正式复现 | 未开始 | 缺少 BraTS/Shanghai/Brisc2025 |
| FedGH 4 客户端正式复现 | 未开始 | 缺少 BraTS/Shanghai/Brisc2025 |
| FedProto 4 客户端正式复现 | 未开始 | 缺少 BraTS/Shanghai/Brisc2025 |
| FedTGP 4 客户端正式复现 | 未开始 | 缺少 BraTS/Shanghai/Brisc2025 |
| FedMM 4 客户端正式复现 | 未开始 | 缺少 BraTS/Shanghai/Brisc2025 |
| FedAMM 4 客户端正式复现 | 未开始 | 缺少 BraTS/Shanghai/Brisc2025，历史 JSON 缺 test 结果 |
| FedMFG 4 客户端正式复现 | 未开始 | 缺少 BraTS/Shanghai/Brisc2025，历史 JSON 缺 test 结果 |

## 消融实验状态

| 消融变体 | 脚本支持 | 正式结果 |
| --- | --- | --- |
| Full FedMFG | 已支持 | 未跑 |
| w/o modality gate | 已支持 | 未跑 |
| w/o combo prototype | 已支持 | 未跑 |
| w/o teacher prototype | 已支持 | 未跑 |
| w/o prototype alignment loss | 已支持 | 未跑 |
| w/o head calibration loss | 已支持 | 未跑 |
| uniform head aggregation | 已支持 | 未跑 |

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

## 当前论文风险

- 没有完整 4 客户端数据时，无法复现毕业论文中的 `FedMFG=90.25%`。
- `FedMFG` 和 `FedAMM` 历史 JSON 缺少正式 test 指标，需要重新跑。
- 正式投稿需要多随机种子，否则结果说服力不足。
