#!/bin/bash
L=/workspace/clear-ec-research/experiments/lane_gpu1_20261106
date -u
ls -la --time-style=full-iso $L/e2e/cropopt_both_f4/f4_s47 | cut -c30-
cat $L/e2e/cropopt_both_f4/f4_s47/exit.json 2>/dev/null
ls -la --time-style=full-iso $L/ssl/cropopt_both_f4/regression_input/run | cut -c30-
ls -la --time-style=full-iso $L/ssl/cropopt_both_f4/regression_input/manifest.json | cut -c30-
ls $L/e2e/cropopt_both_f4/f4_s47/runs/fold4/cpdino_vitb 2>/dev/null | head -5
pgrep -af 'run_e2e|recipeA_ssl_arm' | cut -c1-170
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
tail -1 $L/logs/cropopt_both_f0.log | cut -c1-80
grep -c . $L/../lane_gpu2_20261106/jobs.json 2>/dev/null; grep -n 'run_e2e' $L/../lane_gpu2_20261106/jobs.json 2>/dev/null | cut -c1-200
ls -t $L/../lane_gpu2_20261106/logs | head -4
