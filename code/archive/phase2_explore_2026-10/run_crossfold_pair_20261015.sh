#!/usr/bin/env bash
set -euo pipefail
cd /workspace/clear-ec-research
base=experiments/phase2_crossfold_20261015
split=experiments/target_fold4_20260912/preflight/fold4.npz
runner="$base/source/phase2_crossfold_regression_20261015.py"
common=(--split "$split" --fold 4 --seed 42 --ssl-root "$base/fold4/ssl5" --epochs 20 --schedule-epochs 20 --deadline-utc 2026-10-19T03:59:00Z)
config() {
  case "$1" in
    head1e3_clip1) armargs=(--root "$base/fold4/arms/$1" --head-lr 1e-3 --clip 1) ;;
    head1e4_clip3) armargs=(--root "$base/fold4/arms/$1" --head-lr 1e-4 --clip 3) ;;
    *) exit 2 ;;
  esac
  frozen="$base/fold4/arms/$1/source/phase2_crossfold_regression_20261015.py"
}
for arm in head1e3_clip1 head1e4_clip3; do
  config "$arm"
  CUDA_VISIBLE_DEVICES= /workspace/venv/bin/python -u "$runner" prepare "${common[@]}" "${armargs[@]}"
  CUDA_VISIBLE_DEVICES= /workspace/venv/bin/python -u "$frozen" cpu_tests "${common[@]}" "${armargs[@]}"
done
for arm in head1e3_clip1 head1e4_clip3; do
  config "$arm"
  /workspace/venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 "$frozen" smoke "${common[@]}" "${armargs[@]}"
done
for arm in head1e3_clip1 head1e4_clip3; do
  config "$arm"
  /workspace/venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 "$frozen" train "${common[@]}" "${armargs[@]}"
  CUDA_VISIBLE_DEVICES= /workspace/venv/bin/python -u "$frozen" audit "${common[@]}" "${armargs[@]}"
done
CUDA_VISIBLE_DEVICES= /workspace/venv/bin/python -u "$base/source/phase2_crossfold_pair_audit_20261015.py" \
  --baseline "$base/fold4/arms/head1e3_clip1" \
  --candidate "$base/fold4/arms/head1e4_clip3" \
  --output "$base/fold4/comparison.json"
