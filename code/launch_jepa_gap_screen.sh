#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs experiments/oof_h200
python_bin="${PYTHON_BIN:-/workspace/venv/bin/python}"
encoder_init="experiments/domain_jepa_small/encoder_latest.pt"

common=(
  ROOT=datasets/clear_ec_official OUT_ROOT=experiments/oof_h200
  CPSAM_MODEL=cpsam_models/cpsam_v2 ENCODER_INIT="${encoder_init}"
  NFOLDS=5 FOLD_SEED=2026 FOLD=0 YSRC=slidesmooth
  EVAL_D8=0 FINAL_D8=1 EPOCHS=25 CROP=256 K_CROPS=1
  FEATURE_AGG=0 EVAL_IMAGE_BS=8 POOL=mean LOSS_MODE=multitask BS=48
)

launch() {
  local gpu="$1" name="$2"
  shift 2
  echo "launching ${name} on GPU ${gpu}"
  env CUDA_VISIBLE_DEVICES="${gpu}" "${common[@]}" "$@" \
    "${python_bin}" code/train_oof_h200.py >"logs/${name}.log" 2>&1 &
  echo "$!" >"logs/${name}.pid"
}

launch 0 jepa_small_seed202_f0 TAG=jepa_small_seed202 SEED=202
launch 1 jepa_small_seed303_f0 TAG=jepa_small_seed303 SEED=303
launch 2 jepa_small_seed404_f0 TAG=jepa_small_seed404 SEED=404
launch 3 jepa_small_seed505_f0 TAG=jepa_small_seed505 SEED=505
launch 4 jepa_small_enclr5e6_f0 TAG=jepa_small_enclr5e6 SEED=42 ENC_LR=5e-6
launch 5 jepa_small_enclr2e5_f0 TAG=jepa_small_enclr2e5 SEED=42 ENC_LR=2e-5
launch 6 jepa_small_headlr5e4_f0 TAG=jepa_small_headlr5e4 SEED=42 HEAD_LR=5e-4
launch 7 jepa_small_headlr2e3_f0 TAG=jepa_small_headlr2e3 SEED=42 HEAD_LR=2e-3

wait
"${python_bin}" code/summarize_oof_h200.py >logs/jepa_gap_screen_summary.log 2>&1
echo "JEPA gap screen complete"
