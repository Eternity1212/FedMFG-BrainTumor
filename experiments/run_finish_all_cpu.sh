#!/usr/bin/env bash
# CPU 一条龙：把"多 seed 主对比实验"补齐并汇总。自排队，避免和正在跑的任务抢 CPU。
#   阶段0  等待正在跑的 lowregmid confirm（seed43/seed44）完成；
#   阶段1  跑 seed43/44 的全部基线（local/fedgh/fedproto/fedtgp/fedmm/fedamm），配置与 seed42 完全一致；
#   阶段2  汇总 lowregmid(42/43/44) + 基线(42/43/44) 出 mean±std 主表（report_final）。
# 全程容错：单个算法失败只告警，不中断。日志见 /tmp/finish_all_cpu.log。
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
LOWREG_H="${ROOT_DIR}/paper_outputs/mfg_lowreg/histories"
PUB_H="${ROOT_DIR}/paper_outputs/public_4client/histories"
CMP_DIR="${ROOT_DIR}/paper_outputs/final_compare"
ROUNDS="${ROUNDS:-10}"
BASE_SEEDS="${BASE_SEEDS:-43 44}"
BASE_ALGOS="${BASE_ALGOS:-local fedgh fedproto fedtgp fedmm fedamm}"

ts() { date '+%F %T'; }

rounds_of() { # file -> 已完成轮数（不存在则 -1）
  "${PYTHON_BIN}" - "$1" <<'PY'
import json,os,sys
f=sys.argv[1]
if not os.path.exists(f): print(-1); raise SystemExit
try:
    d=json.load(open(f)); print(len(d.get('val_accuracy',[])))
except Exception: print(-1)
PY
}
is_done() { # file -> 0 if rounds>=ROUNDS or early_stopped
  "${PYTHON_BIN}" - "$1" "${ROUNDS}" <<'PY'
import json,os,sys
f,need=sys.argv[1],int(sys.argv[2])
if not os.path.exists(f): sys.exit(1)
try: d=json.load(open(f))
except Exception: sys.exit(1)
ok = len(d.get('val_accuracy',[]))>=need or d.get('early_stopped') is True or d.get('final_test_accuracy') is not None
sys.exit(0 if ok else 1)
PY
}

echo "[finish_all $(ts)] 阶段0：等待 lowregmid confirm (seed43/seed44) 完成 ..."
for _ in $(seq 1 240); do   # 最多等 ~20h
  s43="${LOWREG_H}/fedmfg-lowregmid_seed43_history.json"
  s44="${LOWREG_H}/fedmfg-lowregmid_seed44_history.json"
  r43=$(rounds_of "$s43"); r44=$(rounds_of "$s44")
  echo "[finish_all $(ts)] 等待中 seed43=${r43}/${ROUNDS} seed44=${r44}/${ROUNDS}"
  if is_done "$s43" && is_done "$s44"; then
    echo "[finish_all $(ts)] lowregmid confirm 已完成。"
    break
  fi
  sleep 300
done

echo "[finish_all $(ts)] 阶段1：补跑 seed ${BASE_SEEDS} 基线 (${BASE_ALGOS}) ..."
for seed in ${BASE_SEEDS}; do
  echo "[finish_all $(ts)] >>> 基线 seed=${seed}"
  SEED="${seed}" ROUNDS="${ROUNDS}" MAX_SAMPLES=0 ALGOS="${BASE_ALGOS}" PYTHON="${PYTHON_BIN}" \
    bash "${ROOT_DIR}/experiments/run_public_4client_baselines.sh" \
    || echo "[finish_all $(ts)] [WARN] 基线 seed=${seed} 出现失败（已容错继续）"
done

echo "[finish_all $(ts)] 阶段2：汇总 mean±std 主表 ..."
rm -rf "${CMP_DIR}"; mkdir -p "${CMP_DIR}"
# 基线：三个 seed
cp "${PUB_H}"/local_seed*_history.json    "${CMP_DIR}/" 2>/dev/null || true
cp "${PUB_H}"/fedgh_seed*_history.json     "${CMP_DIR}/" 2>/dev/null || true
cp "${PUB_H}"/fedproto_seed*_history.json  "${CMP_DIR}/" 2>/dev/null || true
cp "${PUB_H}"/fedtgp_seed*_history.json    "${CMP_DIR}/" 2>/dev/null || true
cp "${PUB_H}"/fedmm_seed*_history.json     "${CMP_DIR}/" 2>/dev/null || true
cp "${PUB_H}"/fedamm_seed*_history.json    "${CMP_DIR}/" 2>/dev/null || true
# 我们的方法：lowregmid 三个 seed
cp "${LOWREG_H}"/fedmfg-lowregmid_seed*_history.json "${CMP_DIR}/" 2>/dev/null || true

"${PYTHON_BIN}" "${ROOT_DIR}/paper_tools/report_final.py" \
  --history_dir "${CMP_DIR}" \
  --output_csv "${CMP_DIR}/final_main_table.csv" | tee "${CMP_DIR}/final_main_table.txt"

echo "[finish_all $(ts)] FINISH_ALL_DONE  -> ${CMP_DIR}/final_main_table.csv"
