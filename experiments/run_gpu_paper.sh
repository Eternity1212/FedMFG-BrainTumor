#!/usr/bin/env bash
# ============================================================================
# FedMFG 论文级 GPU 全分辨率实验（turnkey）。
#
# 一条命令完成 4 个阶段（可用 DO_* 开关单独跑）：
#   STAGE 1 主对比      : 全分辨率 + 多 seed，所有方法。FedMFG 用获胜配置 lowregmid。
#   STAGE 2 基线公平调参: 给 FedProto/FedAMM/FedTGP 同等超参网格(seed42)，避免"只调自己"。
#   STAGE 3 缺失模态曲线: FedMFG vs 基线，在测试期 0/25/50/75% 缺失下的鲁棒性（核心卖点）。
#   STAGE 4 消融        : 关掉 teacher / combo-prototype / modality-gate，证明每个组件有用。
#   STAGE 5 汇总自检    : 双口径报告 + 多 seed 均值方差。
#
# 与 CPU 脚本的区别：不降分辨率（dataset.py 默认全分辨率），开 AMP，batch 更大。
#
# 前提：CUDA + PyTorch(GPU) + MONAI + 依赖；全分辨率数据已在 data/processed。
#       （数据准备见 README / run.sh 的 STAGE 2，或自行预处理到 data/processed）
#
# 用法：
#   bash experiments/run_gpu_paper.sh
#   SEEDS="42 43 44" ROUNDS=16 bash experiments/run_gpu_paper.sh
#   # 显存紧张就调小 batch：
#   CBS_MAP="BraTS=2 Shanghai=4 Figshare=32 Brisc2025=32" bash experiments/run_gpu_paper.sh
#   # 只跑某些阶段：
#   DO_BASETUNE=0 DO_ABLATION=0 bash experiments/run_gpu_paper.sh
# ============================================================================
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CODE_DIR="${ROOT_DIR}/Graduation-Design-main"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/data/processed}"
OUT="${OUT:-${ROOT_DIR}/paper_outputs/gpu_paper}"
PYTHON_BIN="${PYTHON:-python3}"

DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-42 43 44}"
ROUNDS="${ROUNDS:-16}"
LR="${LR:-3e-4}"   # 关键：lowregmid 获胜配置在 CPU 上用的就是 3e-4；1e-3 会训练不稳定（loss 飙到 15+），导致 FedMFG 反而跑不出优势。
CBS_MAP="${CBS_MAP:-BraTS=4 Shanghai=8 Figshare=64 Brisc2025=64}"
ALGOS="${ALGOS:-fedmfg fedamm fedmm fedtgp fedproto fedgh local}"
TEST_RATES="${TEST_RATES:-0.0 0.25 0.5 0.75}"
MISSING_SEED="${MISSING_SEED:-42}"
TUNE_SEED="${TUNE_SEED:-42}"

DO_MAIN="${DO_MAIN:-1}"
DO_BASETUNE="${DO_BASETUNE:-1}"
DO_MISSING="${DO_MISSING:-1}"
DO_ABLATION="${DO_ABLATION:-1}"
DO_REPORT="${DO_REPORT:-1}"

export MPLCONFIGDIR="${ROOT_DIR}/.matplotlib"
# 关键：不要 export FDU_*_SHAPE -> dataset.py 保持全分辨率。
unset FDU_BRATS_SHAPE FDU_SHANGHAI_SHAPE FDU_FIGSHARE_SHAPE FDU_BRISC2025_SHAPE 2>/dev/null || true
mkdir -p "${MPLCONFIGDIR}" \
  "${OUT}/main/histories" "${OUT}/main/plots" \
  "${OUT}/missing/histories" "${OUT}/missing/plots" \
  "${OUT}/ablation/histories" "${OUT}/ablation/plots"

cd "${CODE_DIR}"
log() { echo -e "\n[gpu_paper] $*\n"; }

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

# FedMFG 获胜配置（lowregmid：放松正则的甜点；CPU seed42 上 client-macro F1=79.30 反超基线）
FEDMFG_WIN=(
  --model_mode multimodal
  --mfg_proto_lambda 0.05
  --mfg_head_lambda 0.05
  --mfg_teacher_lambda 0.3
  --mfg_proto_momentum 0.7
  --mfg_proto_tau 1.0
  --mfg_teacher_tau 1.0
  --mfg_head_tau 1.0
  --mfg_head_beta 1.0
  --mfg_head_gamma 1.0
  --mfg_head_weight_mode count_rho_eta
  --mfg_head_personal_alpha 0.0
)

# 通用单次运行： run <out_subdir> <name> <seed> <algo> [extra args...]
run() {
  local sub="$1" name="$2" seed="$3" algo="$4"; shift 4
  echo "===== [${sub}] ${name}_seed${seed} (algo=${algo}) ====="
  "${PYTHON_BIN}" train.py \
    "${COMMON_ARGS[@]}" \
    --seed "${seed}" \
    --algo "${algo}" \
    "$@" \
    --save_dir "${OUT}/${sub}/checkpoints/${name}_seed${seed}" \
    --history_path "${OUT}/${sub}/histories/${name}_seed${seed}_history.json" \
    --plot_dir "${OUT}/${sub}/plots/${name}_seed${seed}" \
    || echo "[WARN] ${name}_seed${seed} failed" >&2
}

# 各算法默认参数（基线用各自合理默认；FedMFG 用获胜配置）
run_algo_default() {
  local sub="$1" seed="$2" algo="$3"
  case "${algo}" in
    local)    run "${sub}" local    "${seed}" local    --model_mode auto ;;
    fedgh)    run "${sub}" fedgh    "${seed}" fedgh    --model_mode auto ;;
    fedproto) run "${sub}" fedproto "${seed}" fedproto --model_mode auto --proto_lambda 1.0 ;;
    fedtgp)   run "${sub}" fedtgp   "${seed}" fedtgp   --model_mode auto --proto_lambda 1.0 --server_epochs 3 ;;
    fedmm)    run "${sub}" fedmm    "${seed}" fedmm    --model_mode multimodal ;;
    fedamm)   run "${sub}" fedamm   "${seed}" fedamm   --model_mode multimodal --amm_mb_lambda 1.0 --amm_mc_lambda 1.0 ;;
    fedmfg)   run "${sub}" fedmfg   "${seed}" fedmfg   "${FEDMFG_WIN[@]}" ;;
    *) echo "Unknown algo ${algo}" >&2 ;;
  esac
}

# ---------------- STAGE 1: 主对比（多 seed 全分辨率）----------------
if [[ "${DO_MAIN}" == "1" ]]; then
  log "STAGE 1: 主对比 多 seed=${SEEDS} 全分辨率 rounds=${ROUNDS}"
  for seed in ${SEEDS}; do
    for algo in ${ALGOS}; do
      run_algo_default main "${seed}" "${algo}"
    done
    "${PYTHON_BIN}" "${ROOT_DIR}/paper_tools/summarize_histories.py" \
      --history_dir "${OUT}/main/histories" \
      --output_csv "${OUT}/main/summary_seed${seed}.csv" || true
  done
fi

# ---------------- STAGE 2: 基线公平调参（seed${TUNE_SEED}）----------------
# 给主要竞争对手同等的超参搜索预算，避免"只调 FedMFG 不调基线"的不公平质疑。
if [[ "${DO_BASETUNE}" == "1" ]]; then
  log "STAGE 2: 基线公平调参 seed=${TUNE_SEED}（结果作为额外行，挑 val 最优写入论文）"
  for pl in 0.5 1.0 2.0; do
    run main "fedproto-pl$(echo $pl|tr -d '.')" "${TUNE_SEED}" fedproto --model_mode auto --proto_lambda "${pl}"
  done
  for l in 0.5 2.0; do
    run main "fedamm-l$(echo $l|tr -d '.')" "${TUNE_SEED}" fedamm --model_mode multimodal --amm_mb_lambda "${l}" --amm_mc_lambda "${l}"
  done
  run main "fedtgp-pl05" "${TUNE_SEED}" fedtgp --model_mode auto --proto_lambda 0.5 --server_epochs 3
fi

# ---------------- STAGE 3: 缺失模态鲁棒性曲线（核心卖点）----------------
# 训练端干净，测试端按 TEST_RATES 缺失；同一 MISSING_SEED 保证各方法看到相同缺失测试集。
if [[ "${DO_MISSING}" == "1" ]]; then
  log "STAGE 3: 缺失模态曲线 test_rates=${TEST_RATES} (missing_seed=${MISSING_SEED})"
  for algo in fedmfg fedproto fedgh fedamm; do
    for rate in ${TEST_RATES}; do
      tag="${algo}_m$(echo ${rate}|tr -d '.')"
      if [[ "${algo}" == "fedmfg" ]]; then
        run missing "${tag}" "${TUNE_SEED}" fedmfg "${FEDMFG_WIN[@]}" \
          --missing_modality_rate_test "${rate}" --missing_modality_seed "${MISSING_SEED}"
      elif [[ "${algo}" == "fedproto" ]]; then
        run missing "${tag}" "${TUNE_SEED}" fedproto --model_mode auto --proto_lambda 1.0 \
          --missing_modality_rate_test "${rate}" --missing_modality_seed "${MISSING_SEED}"
      elif [[ "${algo}" == "fedgh" ]]; then
        run missing "${tag}" "${TUNE_SEED}" fedgh --model_mode auto \
          --missing_modality_rate_test "${rate}" --missing_modality_seed "${MISSING_SEED}"
      else
        run missing "${tag}" "${TUNE_SEED}" fedamm --model_mode multimodal --amm_mb_lambda 1.0 --amm_mc_lambda 1.0 \
          --missing_modality_rate_test "${rate}" --missing_modality_seed "${MISSING_SEED}"
      fi
    done
  done
fi

# ---------------- STAGE 4: FedMFG 消融（seed${TUNE_SEED}）----------------
if [[ "${DO_ABLATION}" == "1" ]]; then
  log "STAGE 4: 消融（在获胜配置基础上逐一关闭组件）seed=${TUNE_SEED}"
  run ablation "abl_full"        "${TUNE_SEED}" fedmfg "${FEDMFG_WIN[@]}"
  run ablation "abl_no_teacher"  "${TUNE_SEED}" fedmfg "${FEDMFG_WIN[@]}" --mfg_disable_teacher
  run ablation "abl_no_combo"    "${TUNE_SEED}" fedmfg "${FEDMFG_WIN[@]}" --mfg_disable_combo_prototype
  run ablation "abl_no_gate"     "${TUNE_SEED}" fedmfg "${FEDMFG_WIN[@]}" --mfg_disable_modality_gate
  run ablation "abl_uniform_head" "${TUNE_SEED}" fedmfg "${FEDMFG_WIN[@]}" --mfg_head_weight_mode uniform
fi

# ---------------- STAGE 5: 汇总自检 ----------------
if [[ "${DO_REPORT}" == "1" ]]; then
  log "STAGE 5: 汇总报告（双口径 + 多 seed 均值方差）"
  echo "===== 主对比（双口径）====="
  "${PYTHON_BIN}" "${ROOT_DIR}/paper_tools/report_final.py" \
    --history_dir "${OUT}/main/histories" || true
  "${PYTHON_BIN}" "${ROOT_DIR}/paper_tools/aggregate_multiseed.py" \
    --history_dir "${OUT}/main/histories" \
    --output_csv "${OUT}/main/summary_all_seeds.csv" || true
  echo "===== 缺失模态曲线 ====="
  "${PYTHON_BIN}" "${ROOT_DIR}/paper_tools/report_final.py" \
    --history_dir "${OUT}/missing/histories" || true
  echo "===== 消融 ====="
  "${PYTHON_BIN}" "${ROOT_DIR}/paper_tools/report_final.py" \
    --history_dir "${OUT}/ablation/histories" || true
fi

log "DONE -> ${OUT}  (main / missing / ablation 三个子目录)"
