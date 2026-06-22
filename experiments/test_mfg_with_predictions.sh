#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CODE_DIR="${ROOT_DIR}/Graduation-Design-main"
DATA_DIR="${1:-${ROOT_DIR}/data/processed}"
CHECKPOINT="${2:-${CODE_DIR}/checkpoints/fedmfg/best_checkpoint.pth}"
OUTPUT_JSON="${3:-${ROOT_DIR}/paper_outputs/test/fedmfg_test_predictions.json}"
PYTHON_BIN="${PYTHON:-python3}"
export MPLCONFIGDIR="${ROOT_DIR}/.matplotlib"
mkdir -p "${MPLCONFIGDIR}"

cd "${CODE_DIR}"

"${PYTHON_BIN}" test.py \
  --root_dir "${DATA_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --output_json "${OUTPUT_JSON}" \
  --collect_predictions \
  --seed 42 \
  --client_names BraTS Shanghai Figshare Brisc2025 \
  --global_rounds 50 \
  --eval_gap 1 \
  --local_epochs 3 \
  --local_learning_rate 1e-3 \
  --client_batch_size_map BraTS=8 Shanghai=64 Figshare=128 Brisc2025=128 \
  --val_ratio 0.1 \
  --model_name resnet18 \
  --model_mode multimodal \
  --num_classes 5 \
  --prototype_dim 512 \
  --dropout 0.0 \
  --save_dir checkpoints/fedmfg \
  --algo fedmfg \
  --mfg_proto_lambda 0.1 \
  --mfg_head_lambda 0.1 \
  --mfg_proto_momentum 0.7 \
  --mfg_proto_tau 1.0 \
  --mfg_teacher_lambda 0.7 \
  --mfg_teacher_tau 1.0 \
  --mfg_head_tau 1.0 \
  --mfg_head_beta 1.0 \
  --mfg_head_weight_mode rho \
  --num_workers 8

"${PYTHON_BIN}" "${ROOT_DIR}/paper_tools/plot_confusion_from_predictions.py" \
  --summary_json "${OUTPUT_JSON}" \
  --output_dir "${ROOT_DIR}/paper_outputs/confusion/fedmfg"
