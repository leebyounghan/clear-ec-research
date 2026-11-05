#!/usr/bin/env bash
set -euo pipefail
cd /workspace/clear-ec-research
base=experiments/phase2_crossfold_20261015
source_dir="$base/ssl0_source"
root="$base/fold4/ssl0_control/head1e3_clip1"
ssl5="$base/fold4/arms/head1e3_clip1"
runner="$source_dir/phase2_crossfold_ssl0_regression_20261015.py"
frozen="$root/source/phase2_crossfold_ssl0_regression_20261015.py"
CUDA_VISIBLE_DEVICES= /workspace/venv/bin/python - "$source_dir" <<'PY'
import hashlib, json, sys
from pathlib import Path
directory = Path(sys.argv[1])
inventory = json.loads((directory/'sha256.json').read_text())
for name, expected in inventory.items():
    assert Path(name).name == name
    assert hashlib.sha256((directory/name).read_bytes()).hexdigest() == expected, name
print('SSL0 frozen launch sources verified', flush=True)
PY
common=(--root "$root" --split experiments/target_fold4_20260912/preflight/fold4.npz --fold 4 --seed 42 --head-lr 1e-3 --clip 1 --epochs 20 --schedule-epochs 20 --data-preflight "$base/data_preflight.json" --matched-ssl5-root "$ssl5" --deadline-utc 2026-10-19T03:59:00Z)
CUDA_VISIBLE_DEVICES= /workspace/venv/bin/python -u "$runner" prepare "${common[@]}"
CUDA_VISIBLE_DEVICES= /workspace/venv/bin/python -u "$frozen" cpu_tests "${common[@]}"
/workspace/venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 "$frozen" smoke "${common[@]}"
/workspace/venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=2 "$frozen" train "${common[@]}"
CUDA_VISIBLE_DEVICES= /workspace/venv/bin/python -u "$frozen" audit "${common[@]}"
CUDA_VISIBLE_DEVICES= /workspace/venv/bin/python -u "$source_dir/phase2_crossfold_ssl_ablation_audit_20261015.py" \
  --ssl0-root "$root" --ssl5-root "$ssl5" --output "$base/fold4/ssl_ablation.json"
