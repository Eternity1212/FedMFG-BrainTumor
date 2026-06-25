#!/usr/bin/env bash
# 把 GPU 实验的"结果文件"推回 GitHub，供本地分析。
# 只推很小的 *_history.json + 汇总 CSV + 运行日志（用 -f 绕过 .gitignore），
# 绝不推大权重文件。本地 git pull 后即可用 report_final.py 分析。
#
# 用法（GPU 服务器仓库根目录，实验跑完或跑到一半都可随时同步）：
#   bash experiments/gpu_push_results.sh
set -uo pipefail
shopt -s nullglob

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT_DIR}"

added=0
for p in \
  paper_outputs/gpu_paper/main/histories/*.json \
  paper_outputs/gpu_paper/missing/histories/*.json \
  paper_outputs/gpu_paper/ablation/histories/*.json \
  paper_outputs/gpu_paper/main/*.csv \
  paper_outputs/gpu_paper_run.log
do
  git add -f "$p" 2>/dev/null && added=1
done

if [[ "${added}" == "0" ]]; then
  echo "暂无结果文件可同步（实验可能还没产出 *_history.json）。"
  exit 0
fi

git commit -m "results: gpu_paper outputs $(date +%F_%T)" || { echo "没有新变更。"; exit 0; }
for i in 1 2 3 4 5; do
  git push origin main && { echo "已推送结果到 GitHub。"; break; } || { echo "git push 重试 $i ..."; sleep 4; }
done
