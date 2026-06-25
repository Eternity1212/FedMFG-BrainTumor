#!/usr/bin/env bash
# 缺失模态强度扫描（支柱 1：FedMFG 的主战场）。
#
# 目的：画出"随测试期模态缺失加重，各方法准确率如何退化"的鲁棒性曲线。
# 预期：没有模态感知的 baseline(FedProto/FedGH) 随缺失加重快速退化，
# FedMFG 靠模态门控+教师原型退化更平缓 —— 这正是论文的核心论点。
#
# 协议（默认）：训练端干净(train_rate=0)，测试端按 0/0.25/0.5/0.75 缺失。
#   缺失模式由 --missing_modality_seed 决定，同一 seed 下所有方法看到完全
#   相同的缺失测试集，保证公平。单模态客户端(Figshare/Brisc)不受影响。
#
# 注意：每个 (算法 x 测试缺失率) 都是一次独立 train+eval，CPU 上较慢。
#   可用 ALGOS / TEST_RATES 缩小范围先拿信号，例如：
#   ALGOS="fedmfg fedproto" TEST_RATES="0.0 0.5 0.75" bash experiments/run_missing_modality_sweep.sh
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CODE_DIR="${ROOT_DIR}/Graduation-Design-main"
DATA_DIR="${ROOT_DIR}/data/processed"
OUT_DIR="${ROOT_DIR}/paper_outputs/missing_sweep"
PYTHON_BIN="${PYTHON:-python3}"
SEED="${SEED:-42}"
ROUNDS="${ROUNDS:-10}"
ALGOS="${ALGOS:-fedmfg fedproto fedgh fedamm}"
TEST_RATES="${TEST_RATES:-0.0 0.25 0.5 0.75}"
TRAIN_RATE="${TRAIN_RATE:-0.0}"
MISSING_SEED="${MISSING_SEED:-42}"   # 固定常数 -> 所有 run 共享同一缺失测试集

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
  --num_classes 5
  --prototype_dim 128
  --dropout 0.0
  --server_early_stopping_patience 5
  --server_early_stopping_min_delta 0.0
  --num_workers 0
  --no-amp
)

# $1=algo，其余=算法特定参数（含 --model_mode）
extra_args() {
  case "$1" in
    fedproto) echo "--model_mode auto --proto_lambda 1.0" ;;
    fedgh)    echo "--model_mode auto" ;;
    fedamm)   echo "--model_mode multimodal --amm_mb_lambda 1.0 --amm_mc_lambda 1.0" ;;
    fedmfg)   echo "--model_mode multimodal --mfg_proto_lambda 0.05 --mfg_head_lambda 0.05 --mfg_proto_momentum 0.7 --mfg_proto_tau 1.0 --mfg_teacher_lambda 0.3 --mfg_teacher_tau 1.0 --mfg_head_tau 1.0 --mfg_head_beta 1.0 --mfg_head_gamma 1.0 --mfg_head_weight_mode count_rho_eta --mfg_head_personal_alpha 0.0" ;;
    *) echo "" ;;
  esac
}

for algo in ${ALGOS}; do
  for rate in ${TEST_RATES}; do
    tag="${algo}_m$(echo "${rate}" | tr -d '.')"   # 例如 fedmfg_m05
    echo "===== [missing-sweep] ${tag} train_rate=${TRAIN_RATE} test_rate=${rate} ====="
    # shellcheck disable=SC2046
    "${PYTHON_BIN}" train.py \
      "${COMMON_ARGS[@]}" \
      --algo "${algo}" \
      $(extra_args "${algo}") \
      --missing_modality_rate_train "${TRAIN_RATE}" \
      --missing_modality_rate_test "${rate}" \
      --missing_modality_seed "${MISSING_SEED}" \
      --save_dir "${OUT_DIR}/checkpoints/${tag}_seed${SEED}" \
      --history_path "${OUT_DIR}/histories/${tag}_seed${SEED}_history.json" \
      --plot_dir "${OUT_DIR}/plots/${tag}_seed${SEED}" \
      || echo "[WARN] ${tag} failed" >&2
  done
done

echo "===== [missing-sweep] done. histories -> ${OUT_DIR}/histories ====="
echo "对比：python3 ${ROOT_DIR}/paper_tools/report_final.py --history_dir ${OUT_DIR}/histories"
