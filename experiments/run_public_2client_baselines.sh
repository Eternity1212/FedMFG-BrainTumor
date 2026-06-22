#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CODE_DIR="${ROOT_DIR}/Graduation-Design-main"
DATA_DIR="${1:-${ROOT_DIR}/data/processed}"
OUTPUT_DIR="${ROOT_DIR}/paper_outputs/public_2client"
PYTHON_BIN="${PYTHON:-python3}"
ROUNDS="${ROUNDS:-10}"
MAX_SAMPLES="${MAX_SAMPLES:-300}"
SEED="${SEED:-42}"

export MPLCONFIGDIR="${ROOT_DIR}/.matplotlib"
mkdir -p "${MPLCONFIGDIR}" "${OUTPUT_DIR}/histories" "${OUTPUT_DIR}/plots"

cd "${CODE_DIR}"

COMMON_ARGS=(
  --root_dir "${DATA_DIR}"
  --seed "${SEED}"
  --client_names Figshare Brisc2025
  --max_samples "${MAX_SAMPLES}"
  --global_rounds "${ROUNDS}"
  --eval_gap 1
  --local_epochs 1
  --local_learning_rate 1e-3
  --batch_size 16
  --val_ratio 0.1
  --model_name resnet18
  --model_mode multimodal
  --num_classes 5
  --prototype_dim 128
  --dropout 0.0
  --server_early_stopping_patience 5
  --server_early_stopping_min_delta 0.0
  --num_workers 0
  --no-amp
)

run_algo() {
  local algo="$1"
  shift
  "${PYTHON_BIN}" train.py \
    "${COMMON_ARGS[@]}" \
    --algo "${algo}" \
    --save_dir "${OUTPUT_DIR}/checkpoints/${algo}" \
    --history_path "${OUTPUT_DIR}/histories/${algo}_history.json" \
    --plot_dir "${OUTPUT_DIR}/plots/${algo}" \
    "$@"
}

run_algo local
run_algo fedgh
run_algo fedproto --proto_lambda 1.0
run_algo fedtgp --proto_lambda 1.0 --server_epochs 3
run_algo fedmm
run_algo fedamm --amm_mb_lambda 1.0 --amm_mc_lambda 1.0
run_algo fedmfg \
  --mfg_proto_lambda 0.1 \
  --mfg_head_lambda 0.1 \
  --mfg_proto_momentum 0.7 \
  --mfg_proto_tau 1.0 \
  --mfg_teacher_lambda 0.7 \
  --mfg_teacher_tau 1.0 \
  --mfg_head_tau 1.0 \
  --mfg_head_beta 1.0 \
  --mfg_head_weight_mode rho

"${PYTHON_BIN}" "${ROOT_DIR}/paper_tools/summarize_histories.py" \
  --history_dir "${OUTPUT_DIR}/histories" \
  --output_csv "${OUTPUT_DIR}/summary.csv"
