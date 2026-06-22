#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
SEEDS="${SEEDS:-42 43 44}"
ROUNDS="${ROUNDS:-5}"
MAX_SAMPLES="${MAX_SAMPLES:-40}"
ALGOS="${ALGOS:-fedgh fedproto fedtgp fedmm fedamm fedmfg}"

for seed in ${SEEDS}; do
  echo "===== public 4-client run: seed=${seed}, rounds=${ROUNDS}, max_samples=${MAX_SAMPLES}, algos=${ALGOS} ====="
  SEED="${seed}" ROUNDS="${ROUNDS}" MAX_SAMPLES="${MAX_SAMPLES}" ALGOS="${ALGOS}" PYTHON="${PYTHON_BIN}" \
    bash "${ROOT_DIR}/experiments/run_public_4client_baselines.sh"
done

"${PYTHON_BIN}" "${ROOT_DIR}/paper_tools/summarize_histories.py" \
  --history_dir "${ROOT_DIR}/paper_outputs/public_4client/histories" \
  --output_csv "${ROOT_DIR}/paper_outputs/public_4client/summary_all_seeds.csv"
