#!/usr/bin/env bash
# FedMFG 低正则扫描（seed42，alpha=0 全局 head 不变）。
#
# 背景：alpha 个性化 head 失败（救 2D 但崩 3D）。原版 FedMFG(73.76) 仍是最好的
# FedMFG 配置，软肋在 2D(Figshare/Brisc 低于 FedProto)。新假设：把 2D 拖下去的
# 不是 head 共享本身，而是过强的跨客户端正则——
#   --mfg_teacher_lambda 0.7  (教师原型约束，3D 受益但可能过度约束 2D 单模态)
#   --mfg_proto_lambda 0.1 / --mfg_head_lambda 0.1
# 思路：保持全局 head(alpha=0)，只调低这些正则，看能否松开 2D 而不伤 3D。
#
# 三个档位直击假设：
#   lowreg_mid    : proto0.05 head0.05 teacher0.3   (整体减半多)
#   lowreg_strong : proto0.02 head0.02 teacher0.1   (大幅放松)
#   noteacher     : proto0.1  head0.1  teacher0.0   (单独去掉教师，隔离其影响)
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CODE_DIR="${ROOT_DIR}/Graduation-Design-main"
DATA_DIR="${ROOT_DIR}/data/processed"
OUT_DIR="${ROOT_DIR}/paper_outputs/mfg_lowreg"
PYTHON_BIN="${PYTHON:-python3}"
SEED="${SEED:-42}"
ROUNDS="${ROUNDS:-10}"

export MPLCONFIGDIR="${ROOT_DIR}/.matplotlib"
export FDU_BRATS_SHAPE="${FDU_BRATS_SHAPE:-32,112,112}"
export FDU_SHANGHAI_SHAPE="${FDU_SHANGHAI_SHAPE:-16,112,112}"
export FDU_FIGSHARE_SHAPE="${FDU_FIGSHARE_SHAPE:-128,128}"
export FDU_BRISC2025_SHAPE="${FDU_BRISC2025_SHAPE:-128,128}"
mkdir -p "${MPLCONFIGDIR}" "${OUT_DIR}/histories" "${OUT_DIR}/plots"

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

# $1=tag，其余=覆盖的正则参数（alpha 固定 0=全局 head）
run_variant() {
  local tag="$1"; shift
  local name="fedmfg-${tag}"
  echo "===== [lowreg-sweep] ${name} seed=${SEED} rounds=${ROUNDS} args: $* ====="
  "${PYTHON_BIN}" train.py \
    "${COMMON_ARGS[@]}" \
    --algo fedmfg \
    --mfg_proto_momentum 0.7 \
    --mfg_proto_tau 1.0 \
    --mfg_teacher_tau 1.0 \
    --mfg_head_tau 1.0 \
    --mfg_head_beta 1.0 \
    --mfg_head_gamma 1.0 \
    --mfg_head_weight_mode count_rho_eta \
    --mfg_head_personal_alpha 0.0 \
    "$@" \
    --save_dir "${OUT_DIR}/checkpoints/${name}_seed${SEED}" \
    --history_path "${OUT_DIR}/histories/${name}_seed${SEED}_history.json" \
    --plot_dir "${OUT_DIR}/plots/${name}_seed${SEED}" \
    || echo "[WARN] ${name} failed" >&2
}

run_variant lowregmid    --mfg_proto_lambda 0.05 --mfg_head_lambda 0.05 --mfg_teacher_lambda 0.3
run_variant lowregstrong --mfg_proto_lambda 0.02 --mfg_head_lambda 0.02 --mfg_teacher_lambda 0.1
run_variant noteacher    --mfg_proto_lambda 0.1  --mfg_head_lambda 0.1  --mfg_teacher_lambda 0.0

echo "===== [lowreg-sweep] done. histories -> ${OUT_DIR}/histories ====="
