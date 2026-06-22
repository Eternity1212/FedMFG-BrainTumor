#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CODE_DIR="${ROOT_DIR}/Graduation-Design-main"
DATA_DIR="${1:-${ROOT_DIR}/data/processed}"

cd "${CODE_DIR}"

COMMON_ARGS=(
  --root_dir "${DATA_DIR}"
  --seed 42
  --client_names BraTS Shanghai Figshare Brisc2025
  --global_rounds 50
  --eval_gap 1
  --local_epochs 3
  --local_learning_rate 1e-3
  --client_batch_size_map BraTS=8 Shanghai=64 Figshare=128 Brisc2025=128
  --val_ratio 0.1
  --model_name resnet18
  --model_mode multimodal
  --num_classes 5
  --prototype_dim 512
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
  --server_early_stopping_patience 10
  --server_early_stopping_min_delta 0.0
  --num_workers 8
)

run_variant() {
  local name="$1"
  shift
  python train.py \
    "${COMMON_ARGS[@]}" \
    --save_dir "checkpoints/ablation_${name}" \
    --history_path "paper_outputs/ablation/${name}_history.json" \
    --plot_dir "paper_outputs/ablation/${name}_plots" \
    "$@"
}

run_variant full
run_variant no_modality_gate --mfg_disable_modality_gate
run_variant no_combo_prototype --mfg_disable_combo_prototype
run_variant no_teacher --mfg_disable_teacher
run_variant no_proto_loss --mfg_proto_lambda 0
run_variant no_head_calibration --mfg_head_lambda 0
run_variant uniform_head --mfg_head_weight_mode uniform
