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

# Five leakage-safe OOF folds establish the trustworthy baseline.
for fold in 0 1 2 3 4; do
  launch "${fold}" "baseline_f${fold}" \
    TAG=baseline FOLD="${fold}" SEED=42 CROP=256 POOL=mean LOSS_MODE=multitask BS=48 EVAL_IMAGE_BS=8
done

# Three high-upside CD screens run concurrently on the same validation fold.
launch 5 cd_seed101_f0 \
  TAG=cd_seed101 FOLD=0 SEED=101 CROP=256 K_CROPS=1 POOL=mean LOSS_MODE=cd BS=48 EVAL_IMAGE_BS=8
launch 6 cd_k4_f0 \
  TAG=cd_k4 FOLD=0 SEED=42 CROP=256 K_CROPS=4 FEATURE_AGG=1 POOL=mean LOSS_MODE=cd BS=16 EVAL_IMAGE_BS=4
launch 7 cd_k8_f0 \
  TAG=cd_k8 FOLD=0 SEED=42 CROP=256 K_CROPS=8 FEATURE_AGG=1 POOL=mean LOSS_MODE=cd BS=8 EVAL_IMAGE_BS=4

wait
echo "stage 1 complete"
