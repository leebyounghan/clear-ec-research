#!/bin/bash
# For the PI to run on the main pod (deploy lane, 2026-11-12, D-PKG-006): copies the real-T4 test files to the Colab session abf17a.
# The deploy lane's session may not copy anything to an external service (its permission system classed this as data exfiltration).
# What goes: 6 small code files (sha256 below) and 5 deploy frames as .mha (1.26 MB each, frames 0, 447, 3348, 5980, 8943). Uses the kernel
# channel (colab upload), so it works while the package rsync holds the ssh slot. Optional: START=1 also starts colab_run_all.sh on the runtime
# once /content/t1 is complete (it checks for /content/t1/context/MANIFEST.json and the rsync no longer running).
set -e
cd /workspace/clear-ec-research/experiments/lane_cellroute_20261109/inference/colab
sha256sum colab_gc_wrap.py colab_t4_jobs.py colab_setup.sh colab_run_all.sh pack_hooks_new.py frames/*.mha
echo 'import os; os.makedirs("/content/t4test/frames", exist_ok=True); print("mkdir ok")' | colab exec -s abf17a
for f in colab_gc_wrap.py colab_t4_jobs.py colab_setup.sh colab_run_all.sh pack_hooks_new.py frames/*.mha; do colab upload -s abf17a "$f" "/content/t4test/$f"; done
echo 'import subprocess; print(subprocess.run("ls -la /content/t4test /content/t4test/frames; sha256sum /content/t4test/*.py /content/t4test/*.sh | cut -c1-16", shell=True, capture_output=True, text=True).stdout)' | colab exec -s abf17a
if [ "${START:-0}" = 1 ]; then
  echo 'import subprocess; subprocess.Popen("cd /content/t4test && nohup bash colab_run_all.sh > run.log 2>&1 &", shell=True, start_new_session=True); print("started")' | colab exec -s abf17a
fi
