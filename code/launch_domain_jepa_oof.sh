#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p experiments/oof_h200 logs

if [[ $# -ne 2 ]]; then
  echo "usage: $0 ENCODER_INIT TAG_PREFIX" >&2
  exit 2
fi
encoder_init="$1"
tag_prefix="$2"
python_bin="${PYTHON_BIN:-/workspace/venv/bin/python}"
epochs="${EPOCHS:-25}"

common=(
  ROOT=datasets/clear_ec_official
  OUT_ROOT=experiments/oof_h200
  CPSAM_MODEL=cpsam_models/cpsam_v2
  ENCODER_INIT="${encoder_init}"
  NFOLDS=5
  FOLD_SEED=2026
  YSRC=slidesmooth
  EVAL_D8=0
  FINAL_D8=1
  EPOCHS="${epochs}"
  CROP=256
  K_CROPS=1
  FEATURE_AGG=0
  EVAL_IMAGE_BS=8
)

launch() {
  local gpu="$1"
  local name="$2"
  shift 2
  echo "launching ${name} on GPU ${gpu}"
  env CUDA_VISIBLE_DEVICES="${gpu}" "${common[@]}" "$@" \
    "${python_bin}" code/train_oof_h200.py >"logs/${name}.log" 2>&1 &
  echo "$!" >"logs/${name}.pid"
}

# Full five-fold mean-pool measurement is the decision-grade comparison.
for fold in 0 1 2 3 4; do
  launch "${fold}" "${tag_prefix}_mean_f${fold}" \
    TAG="${tag_prefix}_mean" FOLD="${fold}" SEED=42 POOL=mean LOSS_MODE=multitask BS=48
done

# Use spare GPUs for diversity candidates that can add ensemble value.
launch 5 "${tag_prefix}_attn_f0" \
  TAG="${tag_prefix}_attn" FOLD=0 SEED=42 POOL=attn LOSS_MODE=multitask BS=40
launch 6 "${tag_prefix}_seed101_f0" \
  TAG="${tag_prefix}_seed101" FOLD=0 SEED=101 POOL=mean LOSS_MODE=multitask BS=48
launch 7 "${tag_prefix}_raw_f0" \
  TAG="${tag_prefix}_raw" FOLD=0 SEED=42 POOL=mean LOSS_MODE=multitask YSRC=raw BS=48

wait
"${python_bin}" code/summarize_oof_h200.py >"logs/${tag_prefix}_oof_summary.log" 2>&1
echo "${tag_prefix} OOF stage complete"
