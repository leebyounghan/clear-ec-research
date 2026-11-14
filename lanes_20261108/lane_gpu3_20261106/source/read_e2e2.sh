#!/bin/bash
# Two-fold read of the strong-view SSL end-to-end jobs by lane GPU 3's read tool; the verdict stays with lane GPU 1.
cd /workspace/clear-ec-research
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /workspace/venv/bin/python experiments/lane_gpu3_20261106/source/read_family_20261106.py e2e_cropopt_both_s46 'lane_gpu1_20261106/e2e/cropopt_both_f{f}/f{f}_s46' --folds 0,4 --added 2>&1 | tail -n 4
echo "--- fold 4 encoder"; /workspace/venv/bin/python - <<PY
import json
m=json.load(open('experiments/lane_gpu1_20261106/e2e/cropopt_both_f4/f4_s46/manifest.json'))
print({k:m[k] for k in ('ssl_epochs','ssl_steps','ssl_seed')}, m['ssl_checkpoint_sha'][:8], m['seed'] if 'seed' in m else '')
PY
echo "--- lane 1 reads"; ls -t experiments/lane_gpu1_20261106/reads | head -n 4
