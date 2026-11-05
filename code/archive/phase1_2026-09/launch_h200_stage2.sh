#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p experiments/oof_h200 logs

PYTHON_BIN="${PYTHON_BIN:-/workspace/venv/bin/python}"
COMMON=(
  ROOT=datasets/clear_ec_official
  OUT_ROOT=experiments/oof_h200
  CPSAM_MODEL=cpsam_models/cpsam_v2
  NFOLDS=5
  FOLD_SEED=2026
  YSRC=slidesmooth
  EVAL_D8=0
  FINAL_D8=1
  EPOCHS="${EPOCHS:-25}"
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
  env CUDA_VISIBLE_DEVICES="${gpu}" "${COMMON[@]}" "$@" \
    "${PYTHON_BIN}" code/train_oof_h200.py >"logs/${name}.log" 2>&1 &
  echo "$!" >"logs/${name}.pid"
}

# The known winning CD ensemble is mean-pool + attention-pool; measure attention honestly on all folds.
for fold in 0 1 2 3 4; do
  launch "${fold}" "attn_f${fold}" \
    TAG=attn FOLD="${fold}" SEED=42 POOL=attn LOSS_MODE=multitask BS=40
done

# Re-run the one baseline fold whose two HEX=0 labels distorted checkpoint selection.
launch 5 baseline_f2_fix \
  TAG=baseline FOLD=2 SEED=42 POOL=mean LOSS_MODE=multitask BS=48

# Screen true multitask seed diversity; CD-only and multi-crop were worse on fold 0.
launch 6 baseline_seed101_f0 \
  TAG=baseline_seed101 FOLD=0 SEED=101 POOL=mean LOSS_MODE=multitask BS=48
launch 7 baseline_seed202_f0 \
  TAG=baseline_seed202 FOLD=0 SEED=202 POOL=mean LOSS_MODE=multitask BS=48

wait
"${PYTHON_BIN}" code/summarize_oof_h200.py >logs/oof_summary.log 2>&1
echo "stage 2 complete"
