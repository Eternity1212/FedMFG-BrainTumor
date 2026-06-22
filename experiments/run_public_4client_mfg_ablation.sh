#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CODE_DIR="${ROOT_DIR}/Graduation-Design-main"
DATA_DIR="${1:-${ROOT_DIR}/data/processed}"
OUTPUT_DIR="${ROOT_DIR}/paper_outputs/public_4client_ablation"
PYTHON_BIN="${PYTHON:-python3}"
ROUNDS="${ROUNDS:-12}"
SEED="${SEED:-42}"

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
  --local_learning_rate 1e-3
  --batch_size 16
  --client_batch_size_map BraTS=2 Shanghai=4 Figshare=32 Brisc2025=32
  --val_ratio 0.1
  --model_name resnet18
  --model_mode multimodal
  --num_classes 5
  --prototype_dim 128
  --dropout 0.0
  --algo fedmfg
  --mfg_proto_lambda 0.1
  --mfg_head_lambda 0.1
  --mfg_proto_momentum 0.7
  --mfg_proto_tau 1.0
  --mfg_teacher_lambda 0.7
  --mfg_teacher_tau 1.0
  --mfg_head_tau 1.0
  --mfg_head_beta 1.0
  --mfg_head_weight_mode rho
  --server_early_stopping_patience 5
  --server_early_stopping_min_delta 0.0
  --num_workers 0
  --no-amp
)

run_variant() {
  local name="$1"
  shift
  "${PYTHON_BIN}" train.py \
    "${COMMON_ARGS[@]}" \
    --save_dir "${OUTPUT_DIR}/checkpoints/${name}_seed${SEED}" \
    --history_path "${OUTPUT_DIR}/histories/${name}_seed${SEED}_history.json" \
    --plot_dir "${OUTPUT_DIR}/plots/${name}_seed${SEED}" \
    "$@"
}

run_variant full || echo "[WARN] ablation full failed" >&2
run_variant no_modality_gate --mfg_disable_modality_gate || echo "[WARN] ablation no_modality_gate failed" >&2
run_variant no_combo_prototype --mfg_disable_combo_prototype || echo "[WARN] ablation no_combo_prototype failed" >&2
run_variant no_teacher --mfg_disable_teacher || echo "[WARN] ablation no_teacher failed" >&2
run_variant no_proto_loss --mfg_proto_lambda 0 || echo "[WARN] ablation no_proto_loss failed" >&2
run_variant no_head_calibration --mfg_head_lambda 0 || echo "[WARN] ablation no_head_calibration failed" >&2
run_variant uniform_head --mfg_head_weight_mode uniform || echo "[WARN] ablation uniform_head failed" >&2

"${PYTHON_BIN}" "${ROOT_DIR}/paper_tools/summarize_histories.py" \
  --history_dir "${OUTPUT_DIR}/histories" \
  --output_csv "${OUTPUT_DIR}/summary_seed${SEED}.csv"
