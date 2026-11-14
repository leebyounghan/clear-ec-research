#!/bin/bash
cd /workspace/clear-ec-research
L=experiments/lane_gpu1_20261106; RT=/workspace/clear-ec-research/experiments/phase2_foundation_20261002/runtime
PP=$L/source:experiments/meta_ceiling_20261103/source:experiments/phase2_pooling_20261020/source:$RT
date -u
cat $L/ssl/cropopt_both_f0/run/complete.json; echo
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 PYTHONPATH=$PP /workspace/venv/bin/python $L/source/recipeA_ssl_arm_20261106.py make_input cropopt_both_f0
cat $L/ssl/cropopt_both_f0/regression_input/run/result.json; echo
ls -la --time-style=full-iso $L/ssl/cropopt_both_f0/run/complete.json $L/ssl/cropopt_both_f0/regression_input/run/complete.json | cut -c30-
sha256sum $L/ssl/cropopt_both_f0/regression_input/run/last.pt | cut -c1-16
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 /workspace/venv/bin/python - <<'PY'
import torch
c=torch.load('experiments/lane_gpu1_20261106/ssl/cropopt_both_f0/regression_input/run/last.pt',map_location='cpu',weights_only=True)
d=torch.load('experiments/lane_gpu1_20261106/ssl/cropopt_both_f0/run/epoch05.pt',map_location='cpu',weights_only=True)
same=all(torch.equal(c['encoder'][k],d['encoder'][k]) for k in c['encoder'])
print('epoch',c['epoch'],'arm_epochs',c['arm_epochs'],'step',c['step'],'equals epoch05:',same)
PY
tail -1 $L/logs/cropopt_both_f0.log | cut -c1-80
sed -n 1,10p $L/../lane_gpu3_20261106/jobs.json | cut -c1-200
