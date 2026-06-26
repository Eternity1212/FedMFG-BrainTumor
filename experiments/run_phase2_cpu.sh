#!/usr/bin/env bash
# CPU 第二阶段：在主对比(finish_all)完成后，自动跑论文还需要的两组"效果"：
#   阶段A  缺失模态鲁棒性曲线（核心卖点）：fedmfg(lowregmid) vs fedproto vs fedgh，
#          训练端干净、测试端缺失 0/0.25/0.5/0.75，同一缺失 seed 保证公平。
#   阶段B  剩余消融（基于 lowregmid，seed42）：去 combo 原型 / 去模态门控 / 均匀头聚合。
# 自排队：先等 finish_all 产出 final_compare/final_main_table.csv 再开跑，避免抢 CPU。
# 日志：/tmp/phase2_cpu.log
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CODE_DIR="${ROOT_DIR}/Graduation-Design-main"
DATA_DIR="${ROOT_DIR}/data/processed"
PYTHON_BIN="${PYTHON:-python3}"
ABL_DIR="${ROOT_DIR}/paper_outputs/mfg_lowreg"   # 与 noteacher 同目录，便于一起汇总
SENTINEL="${ROOT_DIR}/paper_outputs/final_compare/final_main_table.csv"
ROUNDS="${ROUNDS:-10}"

ts() { date '+%F %T'; }

echo "[phase2 $(ts)] 阶段0：等待 finish_all 主表完成 (${SENTINEL}) ..."
for _ in $(seq 1 288); do   # 最多等 ~24h
  [ -f "${SENTINEL}" ] && { echo "[phase2 $(ts)] 检测到主表，开始第二阶段。"; break; }
  sleep 300
done

# ---------- 阶段A：缺失模态鲁棒性曲线 ----------
echo "[phase2 $(ts)] 阶段A：缺失模态鲁棒性扫描 ..."
ALGOS="fedmfg fedproto fedgh" TEST_RATES="0.0 0.25 0.5 0.75" SEED=42 ROUNDS="${ROUNDS}" \
  PYTHON="${PYTHON_BIN}" bash "${ROOT_DIR}/experiments/run_missing_modality_sweep.sh" \
  || echo "[phase2 $(ts)] [WARN] 缺失模态扫描部分失败（已容错）"

# ---------- 阶段B：剩余消融 ----------
echo "[phase2 $(ts)] 阶段B：剩余消融 (seed42, lowregmid base) ..."
export MPLCONFIGDIR="${ROOT_DIR}/.matplotlib"
export FDU_BRATS_SHAPE="${FDU_BRATS_SHAPE:-32,112,112}"
export FDU_SHANGHAI_SHAPE="${FDU_SHANGHAI_SHAPE:-16,112,112}"
export FDU_FIGSHARE_SHAPE="${FDU_FIGSHARE_SHAPE:-128,128}"
export FDU_BRISC2025_SHAPE="${FDU_BRISC2025_SHAPE:-128,128}"
mkdir -p "${MPLCONFIGDIR}" "${ABL_DIR}/histories" "${ABL_DIR}/plots"
cd "${CODE_DIR}"

LOWREG_BASE=(
  --root_dir "${DATA_DIR}" --seed 42
  --client_names BraTS Shanghai Figshare Brisc2025
  --global_rounds "${ROUNDS}" --eval_gap 1 --local_epochs 1 --local_learning_rate 3e-4
  --batch_size 16 --client_batch_size_map BraTS=2 Shanghai=4 Figshare=32 Brisc2025=32
  --val_ratio 0.1 --model_name resnet18 --model_mode multimodal --num_classes 5
  --prototype_dim 128 --dropout 0.0
  --server_early_stopping_patience 5 --server_early_stopping_min_delta 0.0
  --num_workers 0 --no-amp --algo fedmfg
  --mfg_proto_lambda 0.05 --mfg_head_lambda 0.05 --mfg_teacher_lambda 0.3
  --mfg_proto_momentum 0.7 --mfg_proto_tau 1.0 --mfg_teacher_tau 1.0
  --mfg_head_tau 1.0 --mfg_head_beta 1.0 --mfg_head_gamma 1.0
  --mfg_head_weight_mode count_rho_eta --mfg_head_personal_alpha 0.0
)

run_abl() { # name  extra-args...
  local name="$1"; shift
  echo "===== [phase2-abl] ${name} ====="
  "${PYTHON_BIN}" train.py "${LOWREG_BASE[@]}" "$@" \
    --save_dir "${ABL_DIR}/checkpoints/${name}_seed42" \
    --history_path "${ABL_DIR}/histories/${name}_seed42_history.json" \
    --plot_dir "${ABL_DIR}/plots/${name}_seed42" \
    || echo "[phase2 $(ts)] [WARN] ${name} failed"
}

run_abl "fedmfg-nocombo"  --mfg_disable_combo_prototype
run_abl "fedmfg-nogate"   --mfg_disable_modality_gate
run_abl "fedmfg-unifhead" --mfg_head_weight_mode uniform

echo "[phase2 $(ts)] 阶段C：汇总缺失模态曲线 ..."
"${PYTHON_BIN}" "${ROOT_DIR}/paper_tools/report_final.py" \
  --history_dir "${ROOT_DIR}/paper_outputs/missing_sweep/histories" \
  --output_csv "${ROOT_DIR}/paper_outputs/missing_sweep/missing_curve.csv" \
  | tee "${ROOT_DIR}/paper_outputs/missing_sweep/missing_curve.txt" || true

echo "[phase2 $(ts)] PHASE2_DONE"
