#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CODE_DIR="${ROOT_DIR}/Graduation-Design-main"
DATA_DIR="${1:-${ROOT_DIR}/data/processed}"
OUTPUT_DIR="${ROOT_DIR}/paper_outputs/public_4client"
PYTHON_BIN="${PYTHON:-python3}"
# MAX_SAMPLES caps samples per client per split. Because samples are grouped by
# label, a small cap biases toward the first classes; leave it as 0 (disabled)
# for class-balanced formal runs, and only set a positive value for quick checks.
ROUNDS="${ROUNDS:-12}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SEED="${SEED:-42}"
ALGOS="${ALGOS:-local fedgh fedproto fedtgp fedmm fedamm fedmfg}"

export MPLCONFIGDIR="${ROOT_DIR}/.matplotlib"
export FDU_BRATS_SHAPE="${FDU_BRATS_SHAPE:-32,112,112}"
export FDU_SHANGHAI_SHAPE="${FDU_SHANGHAI_SHAPE:-16,112,112}"
export FDU_FIGSHARE_SHAPE="${FDU_FIGSHARE_SHAPE:-128,128}"
export FDU_BRISC2025_SHAPE="${FDU_BRISC2025_SHAPE:-128,128}"
mkdir -p "${MPLCONFIGDIR}" "${OUTPUT_DIR}/histories" "${OUTPUT_DIR}/plots"

cd "${CODE_DIR}"

COMMON_ARGS=(
  --root_dir "${DATA_DIR}"
  --seed "${SEED}"
  --client_names BraTS Shanghai Figshare Brisc2025
  --global_rounds "${ROUNDS}"
  --eval_gap 1
  --local_epochs 1
  --local_learning_rate 3e-4
  --batch_size 16
  --client_batch_size_map BraTS=2 Shanghai=4 Figshare=32 Brisc2025=32
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

if [[ "${MAX_SAMPLES}" != "0" ]]; then
  COMMON_ARGS+=(--max_samples "${MAX_SAMPLES}")
fi

run_algo() {
  local algo="$1"
  shift
  "${PYTHON_BIN}" train.py \
    "${COMMON_ARGS[@]}" \
    --algo "${algo}" \
    --save_dir "${OUTPUT_DIR}/checkpoints/${algo}_seed${SEED}" \
    --history_path "${OUTPUT_DIR}/histories/${algo}_seed${SEED}_history.json" \
    --plot_dir "${OUTPUT_DIR}/plots/${algo}_seed${SEED}" \
    "$@"
}

# Fault tolerant: a single algorithm failure logs a warning but does not abort
# the whole (potentially multi-hour, multi-seed) suite.
for algo in ${ALGOS}; do
  case "${algo}" in
    local)
      run_algo local || echo "[WARN] ${algo} (seed ${SEED}) failed" >&2
      ;;
    fedgh)
      run_algo fedgh || echo "[WARN] ${algo} (seed ${SEED}) failed" >&2
      ;;
    fedproto)
      run_algo fedproto --proto_lambda 1.0 || echo "[WARN] ${algo} (seed ${SEED}) failed" >&2
      ;;
    fedtgp)
      run_algo fedtgp --proto_lambda 1.0 --server_epochs 3 || echo "[WARN] ${algo} (seed ${SEED}) failed" >&2
      ;;
    fedmm)
      run_algo fedmm || echo "[WARN] ${algo} (seed ${SEED}) failed" >&2
      ;;
    fedamm)
      run_algo fedamm --amm_mb_lambda 1.0 --amm_mc_lambda 1.0 || echo "[WARN] ${algo} (seed ${SEED}) failed" >&2
      ;;
    fedmfg)
      run_algo fedmfg \
        --mfg_proto_lambda 0.1 \
        --mfg_head_lambda 0.1 \
        --mfg_proto_momentum 0.7 \
        --mfg_proto_tau 1.0 \
        --mfg_teacher_lambda 0.7 \
        --mfg_teacher_tau 1.0 \
        --mfg_head_tau 1.0 \
        --mfg_head_beta 1.0 \
        --mfg_head_weight_mode rho || echo "[WARN] ${algo} (seed ${SEED}) failed" >&2
      ;;
    *)
      echo "Unknown algorithm in ALGOS: ${algo}" >&2
      exit 1
      ;;
  esac
done

"${PYTHON_BIN}" "${ROOT_DIR}/paper_tools/summarize_histories.py" \
  --history_dir "${OUTPUT_DIR}/histories" \
  --output_csv "${OUTPUT_DIR}/summary_seed${SEED}.csv"
