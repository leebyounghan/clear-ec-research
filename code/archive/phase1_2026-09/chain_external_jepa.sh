#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs experiments/domain_jepa_small experiments/domain_jepa_all
python_bin="${PYTHON_BIN:-/workspace/venv/bin/python}"
torchrun=("${python_bin}" -m torch.distributed.run)

echo "[$(date -Is)] waiting for the active supervised OOF jobs"
while pgrep -f '[t]rain_oof_h200.py' >/dev/null; do
  sleep 30
done

echo "[$(date -Is)] phase A: 8-GPU JEPA on CLEAR-EC + verified same/adjacent-modality small sets"
env OMP_NUM_THREADS=4 \
  ROOT=datasets/clear_ec_official \
  EXTERNAL_MANIFEST=datasets/external/ssl_manifest.jsonl \
  OUT=experiments/domain_jepa_small \
  EPOCHS=40 BS=24 EXTERNAL_FRACTION=0.30 MASK_RATIO=0.55 \
  "${torchrun[@]}" --standalone --nproc_per_node=8 code/train_domain_jepa.py \
  >logs/domain_jepa_small.log 2>&1

echo "[$(date -Is)] phase A OOF evaluation"
env EPOCHS=25 bash code/launch_domain_jepa_oof.sh \
  experiments/domain_jepa_small/encoder_latest.pt jepa_small \
  >logs/jepa_small_controller.log 2>&1

echo "[$(date -Is)] waiting for verified FECD flat-mount download"
while pgrep -f '[d]ownload_external_pretrain_data.py' >/dev/null; do
  sleep 60
done
if [[ ! -s datasets/external/zenodo_fecd_flatmount/manifest.json ]]; then
  echo "FECD download manifest is absent; refusing to train on unverified partial files" >&2
  exit 1
fi
"${python_bin}" - <<'PY'
import json
from pathlib import Path
p = Path("datasets/external/zenodo_fecd_flatmount/manifest.json")
d = json.loads(p.read_text())
if d.get("errors") or len(d.get("verified_files", [])) != d.get("expected_files"):
    raise SystemExit("FECD download did not pass complete size/MD5 verification")
print(f"verified {len(d['verified_files'])} FECD TIFFs")
PY

echo "[$(date -Is)] extracting multi-scale high-information FECD tiles"
"${python_bin}" code/build_external_ssl_corpus.py \
  --root datasets/external --tiles-per-scale 16 --workers 2 \
  >logs/build_external_ssl_all.log 2>&1

echo "[$(date -Is)] phase B: continue JEPA with the complete public corpus"
env OMP_NUM_THREADS=4 \
  ROOT=datasets/clear_ec_official \
  EXTERNAL_MANIFEST=datasets/external/ssl_manifest.jsonl \
  OUT=experiments/domain_jepa_all \
  RESUME=experiments/domain_jepa_small/checkpoint_final.pt \
  EPOCHS=55 BS=24 EXTERNAL_FRACTION=0.30 MASK_RATIO=0.55 \
  "${torchrun[@]}" --standalone --nproc_per_node=8 code/train_domain_jepa.py \
  >logs/domain_jepa_all.log 2>&1

echo "[$(date -Is)] phase B OOF evaluation"
env EPOCHS=25 bash code/launch_domain_jepa_oof.sh \
  experiments/domain_jepa_all/encoder_latest.pt jepa_all \
  >logs/jepa_all_controller.log 2>&1

echo "[$(date -Is)] external-domain JEPA pipeline complete"
