#!/usr/bin/env bash
# 在 GPU 服务器上一键后台启动论文实验。
#   - 自动挑选「显存最空」的 GPU，避免和别人的任务挤同一张卡导致 OOM；
#   - 后台 nohup 运行 run_gpu_paper.sh，日志写文件，断开终端也不中断；
#   - 打印 PID 和查看进度的命令。
#
# 用法（在仓库根目录）：
#   bash experiments/gpu_launch.sh
#   # 显存紧张就先调小 batch：
#   CBS_MAP="BraTS=2 Shanghai=4 Figshare=32 Brisc2025=32" bash experiments/gpu_launch.sh
#   # 想指定卡：
#   CUDA_VISIBLE_DEVICES=1 bash experiments/gpu_launch.sh
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG="${ROOT_DIR}/paper_outputs/gpu_paper_run.log"
mkdir -p "${ROOT_DIR}/paper_outputs"

# ---- 自动挑显存最空的 GPU（除非已显式指定 CUDA_VISIBLE_DEVICES）----
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  GPU="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
          | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')"
  if [[ -n "${GPU}" ]]; then
    export CUDA_VISIBLE_DEVICES="${GPU}"
  fi
fi
echo "[gpu_launch] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[gpu_launch] 当前各卡显存："
  nvidia-smi --query-gpu=index,name,memory.free,memory.used --format=csv
fi

echo "[gpu_launch] 后台启动 run_gpu_paper.sh ..."
nohup bash "${ROOT_DIR}/experiments/run_gpu_paper.sh" > "${LOG}" 2>&1 &
PID=$!
echo "[gpu_launch] PID=${PID}"
echo "[gpu_launch] 日志：${LOG}"
echo "[gpu_launch] 看进度：  tail -f ${LOG}"
echo "[gpu_launch] 停止：    kill ${PID}"
