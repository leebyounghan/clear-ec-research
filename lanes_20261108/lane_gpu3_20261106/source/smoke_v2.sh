#!/bin/bash
cd /workspace/clear-ec-research
X=experiments/lane_gpu3_20261106
export CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
for fam in convnext_b32 dinov2_b14; do
  echo "== $fam $(date -u +%T)"
  /workspace/venv/bin/python $X/source/family_recipe_v2_test.py smoke --family $fam --fold 0 --root $X/runs/_smoke/$fam --workers 2 --updates 8 2>&1 | tail -n 6
done
echo "== done $(date -u +%T)"
