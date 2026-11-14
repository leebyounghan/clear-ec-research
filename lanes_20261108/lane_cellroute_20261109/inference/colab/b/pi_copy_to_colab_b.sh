#!/bin/bash
# For the PI to run on the main pod (coordinator, 2026-11-13, D-PKG-007): copies the tier-p package and the real-T4 test files to the PI's
# Colab session b9204d and starts the fp32 / mixed-precision timing jobs there. The coordinator's and the deploy lane's sessions may not copy
# project files to an external service, so this step is the PI's.
# What goes: package_20261112/p (context 567 files, model 443 files, 16.3 GB) by rsync over colab ssh; 4 small scripts and 2 frames (.mha,
# 1.26 MB each) over the kernel channel. Log: /root/xfer_colab_b.log.
set -e
S=b9204d
L=/workspace/clear-ec-research/experiments/lane_cellroute_20261109
cd $L/inference/colab
echo 'import os; os.makedirs("/content/t4test/frames", exist_ok=True); os.makedirs("/content/p", exist_ok=True); print("mkdir ok")' | colab exec -s $S
for f in colab_gc_wrap.py b/colab_t4_jobs_b.py b/colab_setup_p.sh b/colab_run_b.sh frames/00_0.mha frames/04_447.mha; do
  d=/content/t4test/$(echo "$f" | sed 's#^b/##'); colab upload -s $S "$f" "$d"; done
echo "uploads done $(date -u +%H:%M)"
rsync -a --partial -e ssh $L/package_20261112/p/ colabt4b:/content/p/
echo "rsync done $(date -u +%H:%M)"
echo 'import subprocess; subprocess.Popen("cd /content/t4test && nohup bash colab_run_b.sh > run_b.log 2>&1 &", shell=True, start_new_session=True); print("started run_b")' | colab exec -s $S
