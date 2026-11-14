#!/bin/bash
# coordinator 2026-11-13 (D-PKG-007), runs ON the Colab T4 (session b9204d) after the PI's rsync of the tier-p package to /content/p:
# one image per fresh process through gc_entry.py, frame 0, cold disk read each. Log /content/t4test/run_b.log, results out/t4b_*.json.
#   fp16b1    container default on a T4 (fp16, one detector copy): the baseline, expected ~287 s
#   mixhex6   fp16, only hex6_correction in fp32
#   mixkonan  fp16, only deploy_hexkonan_image in fp32
#   mixb1     fp16, both of them (and the green_* modules) in fp32
#   fp32b1    every autocast region fp32
cd /content/t4test
for i in $(seq 1 60); do grep -q ENV_OK /content/setup_env.log && break; sleep 20; done
grep -q ENV_OK /content/setup_env.log || { echo ENV_NOT_READY; exit 1; }
bash colab_setup_p.sh > setup_p.log 2>&1 || { echo SETUP_FAILED; cat setup_p.log; exit 1; }
cat setup_p.log
P=/content/venv/bin/python
$P colab_t4_jobs_b.py fp16b1 t4b_p_fp16b1 1 900
$P colab_t4_jobs_b.py mixhex6 t4b_p_mixhex6 1 1200
$P colab_t4_jobs_b.py mixkonan t4b_p_mixkonan 1 1200
$P colab_t4_jobs_b.py mixb1 t4b_p_mixb1 1 1500
$P colab_t4_jobs_b.py fp32b1 t4b_p_fp32b1 1 1800
echo ALL_DONE_B
