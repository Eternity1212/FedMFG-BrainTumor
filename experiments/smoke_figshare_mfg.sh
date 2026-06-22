#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CODE_DIR="${ROOT_DIR}/Graduation-Design-main"
DATA_DIR="${ROOT_DIR}/data/processed"
OUTPUT_DIR="${ROOT_DIR}/paper_outputs/smoke_figshare_mfg"
PYTHON_BIN="${PYTHON:-python3}"
export MPLCONFIGDIR="${ROOT_DIR}/.matplotlib"
mkdir -p "${MPLCONFIGDIR}"

cd "${CODE_DIR}"

"${PYTHON_BIN}" train.py \
  --root_dir "${DATA_DIR}" \
  --seed 42 \
  --client_names Figshare \
  --max_samples 24 \
  --global_rounds 1 \
  --eval_gap 1 \
  --local_epochs 1 \
  --local_learning_rate 1e-3 \
  --batch_size 4 \
  --val_ratio 0.2 \
  --model_name resnet18 \
  --model_mode multimodal \
  --num_classes 5 \
  --prototype_dim 64 \
  --dropout 0.0 \
  --save_gap 1 \
  --save_total_limit 2 \
  --save_dir "${OUTPUT_DIR}/checkpoints" \
  --history_path "${OUTPUT_DIR}/history.json" \
  --plot_dir "${OUTPUT_DIR}/plots" \
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
  --server_early_stopping_patience 0 \
  --num_workers 0 \
  --no-amp

"${PYTHON_BIN}" test.py \
  --root_dir "${DATA_DIR}" \
  --checkpoint "${OUTPUT_DIR}/checkpoints/checkpoint_round_0.pth" \
  --output_json "${OUTPUT_DIR}/test_predictions.json" \
  --collect_predictions \
  --seed 42 \
  --client_names Figshare \
  --max_samples 24 \
  --global_rounds 1 \
  --eval_gap 1 \
  --local_epochs 1 \
  --local_learning_rate 1e-3 \
  --batch_size 4 \
  --val_ratio 0.2 \
  --model_name resnet18 \
  --model_mode multimodal \
  --num_classes 5 \
  --prototype_dim 64 \
  --dropout 0.0 \
  --save_dir "${OUTPUT_DIR}/checkpoints" \
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
  --num_workers 0 \
  --no-amp

"${PYTHON_BIN}" "${ROOT_DIR}/paper_tools/plot_confusion_from_predictions.py" \
  --summary_json "${OUTPUT_DIR}/test_predictions.json" \
  --output_dir "${OUTPUT_DIR}/confusion"
