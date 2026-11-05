#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
stage1_pid="$(cat logs/stage1_controller.pid)"
echo "waiting for stage 1 controller pid ${stage1_pid}"
while kill -0 "${stage1_pid}" 2>/dev/null; do
  sleep 30
done

/workspace/venv/bin/python code/summarize_oof_h200.py >logs/stage1_summary.log 2>&1
env EPOCHS=25 bash code/launch_h200_stage2.sh >logs/stage2_controller.log 2>&1
