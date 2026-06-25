#!/usr/bin/env bash
# FedMFG 个性化-head alpha 扫描（seed42，与主套件同设置）。
#
# 目标：FedMFG 在客户端宏平均 Macro-F1 上反超 FedProto(77.95)。诊断结论是
# 全局共享 head 抹平了 2D 客户端个性 -> 引入个性化 head：
#   loaded_head = (1-alpha)*global + alpha*local
# alpha 越大越保留本地 head（救 2D），但可能削弱 3D 的跨客户端原型指导。
#
# 各变体写独立 history 名（algo 名带后缀），report_final.py 会作为独立行对比。
# 顺序执行，避免与正在跑的主套件/ personal05 抢满 CPU。
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CODE_DIR="${ROOT_DIR}/Graduation-Design-main"
DATA_DIR="${ROOT_DIR}/data/processed"
OUT_DIR="${ROOT_DIR}/paper_outputs/mfg_alpha"
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

# 运行一个 fedmfg 变体：$1=tag，其余=额外参数
run_variant() {
  local tag="$1"; shift
  local name="fedmfg-${tag}"
  echo "===== [alpha-sweep] ${name} seed=${SEED} rounds=${ROUNDS} args: $* ====="
  "${PYTHON_BIN}" train.py \
    "${COMMON_ARGS[@]}" \
    --algo fedmfg \
    --mfg_proto_momentum 0.7 \
    --mfg_proto_tau 1.0 \
    --mfg_teacher_lambda 0.7 \
    --mfg_teacher_tau 1.0 \
    --mfg_head_tau 1.0 \
    --mfg_head_beta 1.0 \
    --mfg_head_gamma 1.0 \
    --mfg_head_weight_mode count_rho_eta \
    "$@" \
    --save_dir "${OUT_DIR}/checkpoints/${name}_seed${SEED}" \
    --history_path "${OUT_DIR}/histories/${name}_seed${SEED}_history.json" \
    --plot_dir "${OUT_DIR}/plots/${name}_seed${SEED}" \
    || echo "[WARN] ${name} failed" >&2
}

# 默认正则(0.1) + 不同 alpha
run_variant a03      --mfg_proto_lambda 0.1  --mfg_head_lambda 0.1  --mfg_head_personal_alpha 0.3
run_variant a07      --mfg_proto_lambda 0.1  --mfg_head_lambda 0.1  --mfg_head_personal_alpha 0.7
# 更激进救 2D：强个性化 + 低正则
run_variant a07lowreg --mfg_proto_lambda 0.05 --mfg_head_lambda 0.05 --mfg_head_personal_alpha 0.7

echo "===== [alpha-sweep] done. histories -> ${OUT_DIR}/histories ====="
