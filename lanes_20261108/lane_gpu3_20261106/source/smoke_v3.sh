#!/bin/bash
cd /workspace/clear-ec-research
X=experiments/lane_gpu3_20261106
export CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
echo "== cpsam_c512 $(date -u +%T)"
/workspace/venv/bin/python $X/source/family_recipe_v3_test.py smoke --family cpsam_c512 --fold 0 --root $X/runs/_smoke/cpsam_c512 --workers 2 --updates 8 2>&1 | tail -n 5 | cut -c1-560
echo "== done $(date -u +%T)"
