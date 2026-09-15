#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
target="experiments/oof_h200/jepa_small_mean_f4_seed42_c256k1_mean_multitask_slidesmooth/result.json"
echo "[$(date -Is)] waiting for phase-A JEPA OOF completion"
while [[ ! -s "${target}" ]] || pgrep -f '[t]rain_oof_h200.py' >/dev/null; do
  sleep 30
done

controller="$(cat logs/external_jepa_chain.pid)"
if ! kill -0 "${controller}" 2>/dev/null; then
  echo "main controller ${controller} is absent" >&2
  exit 1
fi

# Hold the main chain at its download wait point so phase B cannot contend for
# GPUs if the network finishes unusually early.  Always resume it on exit.
kill -STOP "${controller}"
resume_controller() { kill -CONT "${controller}" 2>/dev/null || true; }
trap resume_controller EXIT INT TERM
echo "[$(date -Is)] main chain paused; launching one-wave JEPA fold-0 screen"
bash code/launch_jepa_gap_screen.sh >logs/jepa_gap_screen_controller.log 2>&1
echo "[$(date -Is)] fold-0 screen finished; resuming main chain"
resume_controller
trap - EXIT INT TERM
