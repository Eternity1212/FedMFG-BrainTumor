# 文献资料索引

本文件夹本地保存了与项目最相关的论文 PDF。为避免公开仓库体积过大和版权风险，PDF 文件默认不提交到 GitHub，只提交本索引文件。

## 已保存 PDF

| 本地文件 | 论文 | 链接 |
| --- | --- | --- |
| `FedAMM_MICCAI2025.pdf` | FedAMM: Federated Learning for Brain Tumor Segmentation with Arbitrary Missing Modalities | https://papers.miccai.org/miccai-2025/paper/1764_paper.pdf |
| `FedMEMA_AAAI2024.pdf` | Federated Modality-Specific Encoders and Multimodal Anchors for Personalized Brain Tumor Segmentation | https://ojs.aaai.org/index.php/AAAI/article/view/27909 |
| `FedMEPD_2026.pdf` | Federated Modality-specific Encoders and Partially Personalized Fusion Decoder for Multimodal Brain Tumor Segmentation | https://arxiv.org/abs/2603.04887 |
| `PmcmFL_TNNLS2026.pdf` | Multimodal Federated Learning with Missing Modality via Prototype Mask and Contrast | https://arxiv.org/abs/2312.13508 |
| `MFCPL_2024.pdf` | Cross-Modal Prototype based Multimodal Federated Learning under Severely Missing Modality | https://arxiv.org/abs/2401.13898 |
| `FIN_Feature_Imputation_2025.pdf` | Multimodal Federated Learning With Missing Modalities through Feature Imputation Network | https://arxiv.org/abs/2505.20232 |
| `MMiC_2025.pdf` | MMiC: Mitigating Modality incompleteness in Multimodal Federated Learning within the Clusters | https://arxiv.org/abs/2505.06911 |

## 阅读优先级

1. 先读 `FedAMM`，它与本项目最接近，决定我们如何写清楚差异。
2. 再读 `FedMEMA/FedMEPD`，学习医学影像联邦缺失模态论文的实验组织方式。
3. 然后读 `PmcmFL/MFCPL`，补充原型学习和缺失模态鲁棒性实验设计。
4. 最后读 `FIN/MMiC`，了解特征补全和聚类式联邦多模态的补充思路。
