#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CODE_DIR="${ROOT_DIR}/Graduation-Design-main"
DATA_DIR="${1:-${ROOT_DIR}/data/processed}"
OUTPUT_DIR="${ROOT_DIR}/paper_outputs/public_4client_smoke_mfg"
PYTHON_BIN="${PYTHON:-python3}"
ROUNDS="${ROUNDS:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-4}"
SEED="${SEED:-42}"

export MPLCONFIGDIR="${ROOT_DIR}/.matplotlib"
export FDU_BRATS_SHAPE="${FDU_BRATS_SHAPE:-32,112,112}"
export FDU_SHANGHAI_SHAPE="${FDU_SHANGHAI_SHAPE:-16,112,112}"
export FDU_FIGSHARE_SHAPE="${FDU_FIGSHARE_SHAPE:-128,128}"
export FDU_BRISC2025_SHAPE="${FDU_BRISC2025_SHAPE:-128,128}"
mkdir -p "${MPLCONFIGDIR}" "${OUTPUT_DIR}"

cd "${CODE_DIR}"

"${PYTHON_BIN}" train.py \
  --root_dir "${DATA_DIR}" \
  --seed "${SEED}" \
  --client_names BraTS Shanghai Figshare Brisc2025 \
  --max_samples "${MAX_SAMPLES}" \
  --global_rounds "${ROUNDS}" \
  --eval_gap 1 \
  --local_epochs 1 \
  --local_learning_rate 1e-3 \
  --batch_size 4 \
  --client_batch_size_map BraTS=1 Shanghai=1 Figshare=4 Brisc2025=4 \
  --val_ratio 0.1 \
  --model_name resnet18 \
  --model_mode multimodal \
  --num_classes 5 \
  --prototype_dim 128 \
  --dropout 0.0 \
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
  --server_early_stopping_patience 3 \
  --server_early_stopping_min_delta 0.0 \
  --num_workers 0 \
  --no-amp \
  --save_dir "${OUTPUT_DIR}/checkpoints" \
  --history_path "${OUTPUT_DIR}/history.json" \
  --plot_dir "${OUTPUT_DIR}/plots"
