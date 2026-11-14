# deploy lane 2026-11-12 (D-PKG-006): the image has /workspace/clear-ec-research as a real directory (COPY code/); the first Colab layout made it a
# symlink, so os.getcwd() after chdir returned /content/t1/context/code and pack_hooks' repo-relative path matching (skip-base, fp16 list,
# digest) missed -> FileNotFoundError on the unshipped base file. Re-lay out as a real directory (cp -a of the staged code tree including the
# model links), check the cwd, then rerun the jobs with the fixed image environment (run3.log).
import subprocess, os
sh = r'''
set -e
W=/workspace/clear-ec-research
pgrep -f "colab_t4_jobs_p[p].py" >/dev/null && { echo "run2 still running"; exit 1; }
[ -L $W ] && rm $W
mkdir -p $W && cp -a /content/t1/context/code/. $W/
cd $W && python3 -c "import os; print('cwd', os.getcwd())"
n=0; miss=0; while read -r rel; do [ -L "$W/$rel" ] || ln -sfn "/opt/ml/model/$rel" "$W/$rel"; [ -e "$W/$rel" ] || miss=$((miss+1)); n=$((n+1)); done < /content/t1/context/model_files.txt
echo "links $n missing $miss"; ls -la $W/experiments/lane_cellroute_20261109/inference/manifests/deploy.json $W/experiments/lane_cellroute_20261109/inference/pack_hooks_staged.py | cut -c1-150
'''
r = subprocess.run(['bash', '-c', sh], capture_output=True, text=True); print(r.stdout[-2000:], r.stderr[-1500:])
if r.returncode == 0 and 'missing 0' in r.stdout:
    runs = [('default', 5), ('fp16b1', 5), ('fp32', 1), ('mix', 1), ('fp32b1', 1), ('bf16', 1)]
    cmd = 'cd /content/t4test && (' + '; '.join(f'/content/venv/bin/python colab_t4_jobs_pp.py {m} t4c3_{m} {n} 900' for m, n in runs) + '; echo ALL_DONE3) > run3.log 2>&1'
    subprocess.Popen(['bash', '-c', cmd], start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); print('started run3')
