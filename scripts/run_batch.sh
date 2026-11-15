#!/usr/bin/env bash
# Run every .mha in a folder, one fresh container per image, and collect predictions.csv.
# Usage: scripts/run_batch.sh <input_dir> <output_dir> [run_case.sh options]
set -euo pipefail
cd "$(dirname "$0")/.."
SRC=${1:?input_dir}; OUT=${2:?output_dir}; shift 2
mkdir -p "$OUT"
for f in "$SRC"/*.mha; do
  n=$(basename "$f" .mha)
  [ -e "$OUT/$n/cell-density.json" ] && { echo "skip $n (done)"; continue; }
  scripts/run_case.sh "$f" "$OUT/$n" "$@" || echo "FAILED $n"
done
python3 scripts/collect_predictions.py "$OUT"
