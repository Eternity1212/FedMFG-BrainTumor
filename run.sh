#!/usr/bin/env bash
###############################################################################
# FedMFG-BrainTumor  ——  GPU 一键复现脚本
#
# 在一台带 GPU 的 Linux 机器上：
#   git clone https://github.com/Eternity1212/FedMFG-BrainTumor.git
#   cd FedMFG-BrainTumor
#   bash run.sh
#
# 这个脚本会顺序完成：
#   STAGE 1  环境安装（venv + CUDA 版 torch + 依赖）
#   STAGE 2  数据下载与全分辨率预处理（Figshare / Brisc2025 / BraTS / Shanghai）
#   STAGE 3  全量主实验（7 种方法 × 多 seed）
#   STAGE 4  消融实验（FedMFG 各模块）
#   STAGE 5  汇总结果（样本加权 + 客户端宏平均/Macro-F1）并自检
#
# 所有阶段都可单独开关与配置（见下方环境变量）。可重复运行，已完成的数据会跳过。
###############################################################################
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${ROOT_DIR}"

# ----------------------------- 可配置项 --------------------------------------
# 阶段开关（1=执行，0=跳过）
DO_ENV="${DO_ENV:-1}"
DO_DATA="${DO_DATA:-1}"
DO_TRAIN="${DO_TRAIN:-1}"
DO_ABLATION="${DO_ABLATION:-1}"
DO_REPORT="${DO_REPORT:-1}"

# 环境
USE_VENV="${USE_VENV:-1}"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
PYTHON_BIN="${PYTHON:-python3}"
# 按你的 CUDA 版本设置；留空则用默认 pip 源（Linux 上通常自带 CUDA 轮子）
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"

# 实验
DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-42 43 44}"
ROUNDS="${ROUNDS:-16}"
ABLATION_SEED="${ABLATION_SEED:-42}"
ABLATION_ROUNDS="${ABLATION_ROUNDS:-16}"
LR="${LR:-1e-3}"
CBS_MAP="${CBS_MAP:-BraTS=4 Shanghai=8 Figshare=64 Brisc2025=64}"

# 数据分辨率与规模（全分辨率对标原文；显存/磁盘紧张可调小）
BRATS_SHAPE="${BRATS_SHAPE:-155,224,224}"
SHANGHAI_SHAPE="${SHANGHAI_SHAPE:-155,224,224}"
BRATS_CASES="${BRATS_CASES:-150}"        # 每类 case 数
SHANGHAI_CASES="${SHANGHAI_CASES:-150}"
BRISC_IMAGE_SIZE="${BRISC_IMAGE_SIZE:-512}"

DATA_DIR="${ROOT_DIR}/data/processed"
OUT_BASE="${ROOT_DIR}/paper_outputs/gpu_fullres"
OUT_ABL="${ROOT_DIR}/paper_outputs/gpu_fullres_ablation"
CLIENTS=(BraTS Shanghai Figshare Brisc2025)

log() { echo -e "\n\033[1;36m[run.sh] $*\033[0m"; }

print_config() {
  log "配置一览"
  cat <<EOF
  设备 DEVICE         = ${DEVICE}
  随机种子 SEEDS      = ${SEEDS}
  通信轮数 ROUNDS     = ${ROUNDS}
  学习率 LR           = ${LR}
  batch 映射          = ${CBS_MAP}
  BraTS 分辨率/case   = ${BRATS_SHAPE} / 每类 ${BRATS_CASES}
  Shanghai 分辨率/case= ${SHANGHAI_SHAPE} / 每类 ${SHANGHAI_CASES}
  Brisc2025 尺寸      = ${BRISC_IMAGE_SIZE}
  阶段开关 env/data/train/ablation/report = ${DO_ENV}/${DO_DATA}/${DO_TRAIN}/${DO_ABLATION}/${DO_REPORT}
EOF
}

###############################################################################
# STAGE 1: 环境
###############################################################################
setup_env() {
  log "STAGE 1: 安装环境"
  if [[ "${USE_VENV}" == "1" ]]; then
    if [[ ! -d "${VENV_DIR}" ]]; then
      "${PYTHON_BIN}" -m venv "${VENV_DIR}"
    fi
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    PYTHON_BIN="python"
  fi
  "${PYTHON_BIN}" -m pip install --upgrade pip
  if [[ -n "${TORCH_INDEX_URL}" ]]; then
    "${PYTHON_BIN}" -m pip install torch --index-url "${TORCH_INDEX_URL}"
  else
    "${PYTHON_BIN}" -m pip install torch
  fi
  "${PYTHON_BIN}" -m pip install -r "${ROOT_DIR}/Graduation-Design-main/requirements.txt"
  "${PYTHON_BIN}" -m pip install nibabel huggingface_hub
  log "GPU 检测"
  "${PYTHON_BIN}" - <<'PY'
import torch
print("torch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("警告：未检测到 CUDA GPU。可设置 DEVICE=cpu，但会非常慢。")
PY
}

###############################################################################
# STAGE 2: 数据
###############################################################################
prepare_data() {
  log "STAGE 2: 数据下载与全分辨率预处理"

  # --- Figshare (2D, 3 类, 原尺寸) ---
  if [[ -d "${DATA_DIR}/Figshare/train" ]]; then
    log "Figshare 已存在，跳过"
  else
    log "预处理 Figshare"
    "${PYTHON_BIN}" data/scripts/preprocess_figshare_hf.py --output_dir "${DATA_DIR}/Figshare" --overwrite
  fi

  # --- Brisc2025 (2D, 4 类, 512x512) ---
  if [[ -d "${DATA_DIR}/Brisc2025/train" ]]; then
    log "Brisc2025 已存在，跳过"
  else
    log "下载 Brisc2025 (Zenodo)"
    bash data/scripts/download_brisc2025_zenodo.sh
    log "预处理 Brisc2025"
    "${PYTHON_BIN}" data/scripts/preprocess_brisc2025.py \
      --zip_path data/raw/brisc2025/brisc2025.zip \
      --output_dir "${DATA_DIR}/Brisc2025" \
      --image_size "${BRISC_IMAGE_SIZE}" --overwrite
  fi

  # --- BraTS + Shanghai (3D, 全分辨率) ---
  if [[ -d "${DATA_DIR}/BraTS/train" && -d "${DATA_DIR}/Shanghai/train" ]]; then
    log "BraTS / Shanghai 已存在，跳过"
  else
    log "下载并预处理 BraTS / Shanghai 3D（全分辨率，可能较慢、占磁盘）"
    "${PYTHON_BIN}" data/scripts/preprocess_brats_3d_hf.py \
      --output_root "${DATA_DIR}" \
      --brats_cases_per_class "${BRATS_CASES}" \
      --shanghai_cases_per_class "${SHANGHAI_CASES}" \
      --brats_shape "${BRATS_SHAPE}" \
      --shanghai_shape "${SHANGHAI_SHAPE}" \
      --test_ratio 0.2 --overwrite
  fi

  log "数据统计"
  mkdir -p "${OUT_BASE}"
  "${PYTHON_BIN}" data/scripts/summarize_dataset.py \
    --processed_dir "${DATA_DIR}" \
    --output_csv "${OUT_BASE}/dataset_summary.csv" || true
}

###############################################################################
# STAGE 3: 主实验（全量，多 seed，多方法对比）
###############################################################################
run_training() {
  log "STAGE 3: 全量主实验（7 方法 × seeds=${SEEDS}）"
  SEEDS="${SEEDS}" ROUNDS="${ROUNDS}" DEVICE="${DEVICE}" LR="${LR}" \
    CBS_MAP="${CBS_MAP}" OUTPUT_DIR="${OUT_BASE}" PYTHON="${PYTHON_BIN}" \
    bash experiments/run_gpu_fullres_suite.sh "${DATA_DIR}"
}

###############################################################################
# STAGE 4: 消融（GPU 全分辨率）
###############################################################################
run_ablation() {
  log "STAGE 4: FedMFG 消融（seed=${ABLATION_SEED}）"
  mkdir -p "${OUT_ABL}/histories" "${OUT_ABL}/plots"
  # 全分辨率：不要设置 FDU_*_SHAPE
  unset FDU_BRATS_SHAPE FDU_SHANGHAI_SHAPE FDU_FIGSHARE_SHAPE FDU_BRISC2025_SHAPE 2>/dev/null || true
  export MPLCONFIGDIR="${ROOT_DIR}/.matplotlib"; mkdir -p "${MPLCONFIGDIR}"
  pushd "${ROOT_DIR}/Graduation-Design-main" >/dev/null

  local COMMON=(
    --root_dir "${DATA_DIR}" --seed "${ABLATION_SEED}"
    --client_names BraTS Shanghai Figshare Brisc2025
    --global_rounds "${ABLATION_ROUNDS}" --eval_gap 1 --local_epochs 1
    --local_learning_rate "${LR}" --batch_size 32
    --client_batch_size_map ${CBS_MAP}
    --val_ratio 0.1 --model_name resnet18 --model_mode multimodal
    --num_classes 5 --prototype_dim 128 --dropout 0.0 --device "${DEVICE}"
    --algo fedmfg
    --mfg_proto_lambda 0.1 --mfg_head_lambda 0.1
    --mfg_proto_momentum 0.7 --mfg_proto_tau 1.0
    --mfg_teacher_lambda 0.7 --mfg_teacher_tau 1.0
    --mfg_head_tau 1.0 --mfg_head_beta 1.0 --mfg_head_gamma 1.0
    --mfg_head_weight_mode count_rho_eta
    --server_early_stopping_patience 6 --server_early_stopping_min_delta 0.0
    --num_workers 4
  )
  variant() {
    local name="$1"; shift
    "${PYTHON_BIN}" train.py "${COMMON[@]}" \
      --save_dir "${OUT_ABL}/checkpoints/${name}_seed${ABLATION_SEED}" \
      --history_path "${OUT_ABL}/histories/${name}_seed${ABLATION_SEED}_history.json" \
      --plot_dir "${OUT_ABL}/plots/${name}_seed${ABLATION_SEED}" "$@" \
      || echo "[WARN] ablation ${name} failed" >&2
  }
  variant full
  variant no_modality_gate     --mfg_disable_modality_gate
  variant no_combo_prototype   --mfg_disable_combo_prototype
  variant no_teacher           --mfg_disable_teacher
  variant no_proto_loss        --mfg_proto_lambda 0
  variant no_head_calibration  --mfg_head_lambda 0
  variant uniform_head         --mfg_head_weight_mode uniform
  variant count_blind_head     --mfg_head_weight_mode rho_eta
  popd >/dev/null
}

###############################################################################
# STAGE 5: 汇总 + 自检
###############################################################################
make_report() {
  log "STAGE 5: 汇总结果"
  mkdir -p "${ROOT_DIR}/paper/results"

  log "主实验：多 seed 汇总（均值±std）"
  "${PYTHON_BIN}" paper_tools/aggregate_multiseed.py \
    --history_dir "${OUT_BASE}/histories" \
    --output_csv "${OUT_BASE}/summary_all_seeds.csv" || true

  log "主实验：双口径报告（样本加权 + 客户端宏平均/Macro-F1，并判定最优方法）"
  "${PYTHON_BIN}" paper_tools/report_final.py \
    --history_dir "${OUT_BASE}/histories" \
    --clients "${CLIENTS[@]}" \
    --output_csv "${ROOT_DIR}/paper/results/gpu_main_report.csv" || true

  if [[ -d "${OUT_ABL}/histories" ]]; then
    log "消融：报告"
    "${PYTHON_BIN}" paper_tools/report_final.py \
      --history_dir "${OUT_ABL}/histories" \
      --clients "${CLIENTS[@]}" \
      --output_csv "${ROOT_DIR}/paper/results/gpu_ablation_report.csv" || true
  fi

  log "结果文件位置"
  cat <<EOF
  主实验逐 seed CSV : ${OUT_BASE}/summary_seed*.csv
  主实验多 seed 汇总: ${OUT_BASE}/summary_all_seeds.csv
  主实验双口径报告  : ${ROOT_DIR}/paper/results/gpu_main_report.csv
  消融报告          : ${ROOT_DIR}/paper/results/gpu_ablation_report.csv
  原始 history(逐轮): ${OUT_BASE}/histories/  与  ${OUT_ABL}/histories/
EOF
}

###############################################################################
# 主流程
###############################################################################
print_config
[[ "${DO_ENV}"      == "1" ]] && setup_env      || log "跳过 STAGE 1 (环境)"
# venv 激活需在后续阶段保持；若 DO_ENV=0 但要用 venv，手动 source 后再跑
[[ "${USE_VENV}" == "1" && -d "${VENV_DIR}" ]] && { source "${VENV_DIR}/bin/activate"; PYTHON_BIN="python"; }
[[ "${DO_DATA}"     == "1" ]] && prepare_data   || log "跳过 STAGE 2 (数据)"
[[ "${DO_TRAIN}"    == "1" ]] && run_training   || log "跳过 STAGE 3 (主实验)"
[[ "${DO_ABLATION}" == "1" ]] && run_ablation   || log "跳过 STAGE 4 (消融)"
[[ "${DO_REPORT}"   == "1" ]] && make_report    || log "跳过 STAGE 5 (汇总)"

log "全部完成 ✅  结果见 paper_outputs/gpu_fullres/ 与 paper/results/"
