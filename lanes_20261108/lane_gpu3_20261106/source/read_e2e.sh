#!/bin/bash
# Read of the end-to-end job by lane GPU 3's read tool. Runs on the pod, time-limited by the caller.
cd /workspace/clear-ec-research
echo "--- fold 4 state"; ls experiments/lane_gpu1_20261106/e2e/cropopt_both_f4/; cat experiments/lane_gpu1_20261106/e2e/cropopt_both_f4/f4_s46/exit.json 2>/dev/null; ls experiments/lane_gpu1_20261106/e2e/cropopt_both_f4/f4_s46/runs/fold4/*/predictions_epoch_*.npz 2>/dev/null | wc -l
echo "--- lane 1 reads"; ls -la --time-style=+%FT%TZ experiments/lane_gpu1_20261106/reads | tail -n 8 | cut -c30-
echo "--- audit"; ls experiments/lane_gpu1_20261106/e2e/cropopt_both_f0/f0_s46/; grep -E "^###" experiments/lane_gpu3_20261106/logs/e2e_cropopt_both_f0.log | tail -n 3
echo "--- read"; OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /workspace/venv/bin/python experiments/lane_gpu3_20261106/source/read_family_20261106.py e2e_cropopt_both_f0_s46 "lane_gpu1_20261106/e2e/cropopt_both_f0/f0_s46" --folds 0 --added 2>&1 | tail -n 6
