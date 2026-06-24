#!/usr/bin/env bash
# 检查点清道夫：周期性删除"陈旧"的大权重文件，防止磁盘被撑满。
#
# 背景：每个 .pth 检查点约 5G（含全部客户端模型+优化器）。论文只需要
# histories/ 里的 JSON 结果，train.py 收尾也不会回读 best_checkpoint，
# 套件不使用 --resume，因此所有 .pth 对结果都是冗余的——可放心删除。
#
# 策略：只删除「最近 KEEP_MIN 分钟内未被修改」的 *.pth。
#   - 正在训练的活跃运行会不断刷新 best_checkpoint.pth → 不会被删；
#   - 已经跑完的算法残留的 best/final 权重很快变陈旧 → 被回收。
# histories/ plots/ summary CSV 等结果文件一律不动。
#
# 用法：
#   nohup bash paper_tools/checkpoint_janitor.sh > paper_outputs/janitor.log 2>&1 &
# 停止：
#   pkill -f checkpoint_janitor.sh

set -u
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${TARGET:-${ROOT_DIR}/paper_outputs}"
KEEP_MIN="${KEEP_MIN:-15}"     # 保留最近多少分钟内更新过的 .pth
INTERVAL="${INTERVAL:-300}"    # 每隔多少秒清理一次

echo "[janitor] start  target=${TARGET}  keep_min=${KEEP_MIN}  interval=${INTERVAL}s"
while true; do
  ts="$(date '+%F %T')"
  freed=0
  # 找出陈旧的大权重并删除（mmin +KEEP_MIN = 超过 KEEP_MIN 分钟未修改）
  while IFS= read -r f; do
    [ -f "$f" ] || continue
    sz=$(stat -f %z "$f" 2>/dev/null || stat -c %s "$f" 2>/dev/null || echo 0)
    rm -f "$f" && freed=$((freed + sz)) && echo "[janitor] ${ts} 删除 $(basename "$(dirname "$f")")/$(basename "$f")"
  done < <(find "${TARGET}" -type f -name '*.pth' -mmin +"${KEEP_MIN}" 2>/dev/null)
  if [ "${freed}" -gt 0 ]; then
    echo "[janitor] ${ts} 本轮释放 $((freed / 1024 / 1024)) MB；当前 paper_outputs=$(du -sh "${TARGET}" 2>/dev/null | cut -f1)"
  fi
  sleep "${INTERVAL}"
done
