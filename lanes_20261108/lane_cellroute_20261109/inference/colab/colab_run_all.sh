#!/bin/bash
# deploy lane 2026-11-12 (D-PKG-006), runs ON the Colab T4 after the package rsync to /content/t1 has finished: image layout, then the jobs in
# priority order (each job = one image, fresh process, gc_entry.py). Log /content/t4test/run.log, results /content/t4test/out/t4c_*.json.
#   default 5 frames  container default on a T4 (staged pack_hooks: fp16 below compute capability 8.0)
#   fp32    1 frame   all-fp32 (job killed at 900 s: past 300 s either way)
#   mix     1 frame   fp32 only in the detector / hex6 modules
#   fp16b1  5 frames  fp16 with one copy instead of 4 identical copies in the 13 detectors
#   bf16    1 frame   native bf16 as written (does torch 2.11 run it on sm75, and how fast)
#   fp32b1  1 frame   all-fp32 with one copy
cd /content/t4test
for i in $(seq 1 240); do pgrep -f 'rsync --server' >/dev/null || break; sleep 30; done   # the PI's package rsync must have finished (max 2 h)
pgrep -f 'rsync --server' >/dev/null && { echo RSYNC_STILL_RUNNING; exit 1; }
bash colab_setup.sh > setup.log 2>&1 || { echo SETUP_FAILED; cat setup.log; exit 1; }
P=/content/venv/bin/python
$P colab_t4_jobs.py default t4c_default 5 900
$P colab_t4_jobs.py fp32 t4c_fp32 1 900
$P colab_t4_jobs.py mix t4c_mix 1 900
$P colab_t4_jobs.py fp16b1 t4c_fp16b1 5 900
$P colab_t4_jobs.py bf16 t4c_bf16 1 900
$P colab_t4_jobs.py fp32b1 t4c_fp32b1 1 900
echo ALL_DONE
