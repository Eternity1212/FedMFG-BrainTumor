#!/usr/bin/env bash
# Full-resolution GPU run that mirrors the thesis setup (8x A100 in the paper,
# but works on a single modern GPU with the batch sizes below). Unlike the
# CPU scripts, this does NOT downsample: it leaves dataset.py at its full-res
# defaults (BraTS 155x224x224, Shanghai full grid, 2D 512x512), enables AMP,
# uses AdamW lr=1e-3 and 16 communication rounds.
#
# Prerequisites:
#   1. A CUDA GPU + PyTorch built with CUDA, MONAI, and repo requirements.
#   2. Full-resolution data prepared under data/processed (see README_GPU.md).
#
# Usage:
#   bash experiments/run_gpu_fullres_suite.sh [DATA_DIR]
#
# Override with env vars, e.g.:
#   SEEDS="42 43 44" ROUNDS=16 bash experiments/run_gpu_fullres_suite.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CODE_DIR="${ROOT_DIR}/Graduation-Design-main"
DATA_DIR="${1:-${ROOT_DIR}/data/processed}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/paper_outputs/gpu_fullres}"
PYTHON_BIN="${PYTHON:-python3}"

DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-42 43 44}"
ROUNDS="${ROUNDS:-16}"
LR="${LR:-1e-3}"
# Full-res 3D is memory heavy; tune to your GPU. 2D can use large batches.
CBS_MAP="${CBS_MAP:-BraTS=4 Shanghai=8 Figshare=64 Brisc2025=64}"
ALGOS="${ALGOS:-fedmfg fedamm fedmm fedtgp fedproto fedgh local}"

export MPLCONFIGDIR="${ROOT_DIR}/.matplotlib"
# IMPORTANT: do NOT export FDU_*_SHAPE here -> dataset.py keeps full resolution.
unset FDU_BRATS_SHAPE FDU_SHANGHAI_SHAPE FDU_FIGSHARE_SHAPE FDU_BRISC2025_SHAPE 2>/dev/null || true
mkdir -p "${MPLCONFIGDIR}" "${OUTPUT_DIR}/histories" "${OUTPUT_DIR}/plots"

cd "${CODE_DIR}"

COMMON_ARGS=(
  --root_dir "${DATA_DIR}"
  --client_names BraTS Shanghai Figshare Brisc2025
  --global_rounds "${ROUNDS}"
  --eval_gap 1
  --local_epochs 1
  --local_learning_rate "${LR}"
  --batch_size 32
  --client_batch_size_map ${CBS_MAP}
  --val_ratio 0.1
  --model_name resnet18
  --num_classes 5
  --prototype_dim 128
  --dropout 0.0
  --device "${DEVICE}"
  --server_early_stopping_patience 6
  --server_early_stopping_min_delta 0.0
  --num_workers 4
)

run_algo() {
  local seed="$1" algo="$2"; shift 2
  "${PYTHON_BIN}" train.py \
    "${COMMON_ARGS[@]}" \
    --seed "${seed}" \
    --algo "${algo}" \
    --save_dir "${OUTPUT_DIR}/checkpoints/${algo}_seed${seed}" \
    --history_path "${OUTPUT_DIR}/histories/${algo}_seed${seed}_history.json" \
    --plot_dir "${OUTPUT_DIR}/plots/${algo}_seed${seed}" \
    "$@"
}

for seed in ${SEEDS}; do
  echo "===== GPU full-res run: seed=${seed}, rounds=${ROUNDS}, lr=${LR} ====="
  for algo in ${ALGOS}; do
    case "${algo}" in
      local)    run_algo "${seed}" local    --model_mode auto || echo "[WARN] local s${seed} failed" >&2 ;;
      fedgh)    run_algo "${seed}" fedgh    --model_mode auto || echo "[WARN] fedgh s${seed} failed" >&2 ;;
      fedproto) run_algo "${seed}" fedproto --model_mode auto --proto_lambda 1.0 || echo "[WARN] fedproto s${seed} failed" >&2 ;;
      fedtgp)   run_algo "${seed}" fedtgp   --model_mode auto --proto_lambda 1.0 --server_epochs 3 || echo "[WARN] fedtgp s${seed} failed" >&2 ;;
      fedmm)    run_algo "${seed}" fedmm    --model_mode multimodal || echo "[WARN] fedmm s${seed} failed" >&2 ;;
      fedamm)   run_algo "${seed}" fedamm   --model_mode multimodal --amm_mb_lambda 1.0 --amm_mc_lambda 1.0 || echo "[WARN] fedamm s${seed} failed" >&2 ;;
      fedmfg)   run_algo "${seed}" fedmfg   --model_mode multimodal \
                  --mfg_proto_lambda 0.1 --mfg_head_lambda 0.1 \
                  --mfg_proto_momentum 0.7 --mfg_proto_tau 1.0 \
                  --mfg_teacher_lambda 0.7 --mfg_teacher_tau 1.0 \
                  --mfg_head_tau 1.0 --mfg_head_beta 1.0 --mfg_head_gamma 1.0 \
                  --mfg_head_weight_mode count_rho_eta || echo "[WARN] fedmfg s${seed} failed" >&2 ;;
      *) echo "Unknown algo ${algo}" >&2; exit 1 ;;
    esac
  done
  "${PYTHON_BIN}" "${ROOT_DIR}/paper_tools/summarize_histories.py" \
    --history_dir "${OUTPUT_DIR}/histories" \
    --output_csv "${OUTPUT_DIR}/summary_seed${seed}.csv" || true
done

"${PYTHON_BIN}" "${ROOT_DIR}/paper_tools/aggregate_multiseed.py" \
  --history_dir "${OUTPUT_DIR}/histories" \
  --output_csv "${OUTPUT_DIR}/summary_all_seeds.csv" || true

echo "########## DONE: GPU full-res suite -> ${OUTPUT_DIR} ##########"
