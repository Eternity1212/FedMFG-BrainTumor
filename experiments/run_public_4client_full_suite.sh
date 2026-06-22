#!/usr/bin/env bash
# Run the full public 4-client experiment suite sequentially:
#   1. Multi-seed baselines (all algorithms) for mean/std main table.
#   2. FedMFG ablation (seed 42) for the ablation table.
# Designed to run as a single long background job on a CPU-only machine.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
SEEDS="${SEEDS:-42 43 44}"
ROUNDS="${ROUNDS:-12}"
ABLATION_SEED="${ABLATION_SEED:-42}"
ABLATION_ROUNDS="${ABLATION_ROUNDS:-12}"
ALGOS="${ALGOS:-fedmfg fedamm fedmm fedtgp fedproto fedgh local}"

echo "########## STAGE 1: multi-seed baselines ##########"
SEEDS="${SEEDS}" ROUNDS="${ROUNDS}" MAX_SAMPLES=0 ALGOS="${ALGOS}" PYTHON="${PYTHON_BIN}" \
  bash "${ROOT_DIR}/experiments/run_public_4client_multiseed.sh"

echo "########## STAGE 2: FedMFG ablation (seed ${ABLATION_SEED}) ##########"
SEED="${ABLATION_SEED}" ROUNDS="${ABLATION_ROUNDS}" PYTHON="${PYTHON_BIN}" \
  bash "${ROOT_DIR}/experiments/run_public_4client_mfg_ablation.sh"

echo "########## DONE: full public 4-client suite ##########"
echo "Baseline per-seed CSVs : ${ROOT_DIR}/paper_outputs/public_4client/summary_seed*.csv"
echo "Baseline all-seed CSV  : ${ROOT_DIR}/paper_outputs/public_4client/summary_all_seeds.csv"
echo "Ablation CSV           : ${ROOT_DIR}/paper_outputs/public_4client_ablation/summary_seed${ABLATION_SEED}.csv"
