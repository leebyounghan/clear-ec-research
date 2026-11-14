# deploy lane 2026-11-12 (D-PKG-006): stop run4 (CUDA OOM at the attention-pool stage: CLEAR_EC_UNLOAD was not set in the image, so every stage's
# models stayed on the 15 GB T4); derive colab_t4_jobs_pp2.py = the fixed image environment incl. CLEAR_EC_UNLOAD=1; rerun (run5.log).
import subprocess, time
subprocess.run('pkill -f "run[4].log"; sleep 1; pkill -f "colab_t4_jobs_p[p].py"; sleep 1; pkill -f "colab_gc_wra[p].py"', shell=True); time.sleep(5)
T = '/content/t4test/'; s = open(T + 'colab_t4_jobs_pp.py').read()
s = s.replace("PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',", "PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', CLEAR_EC_UNLOAD='1',")
assert "CLEAR_EC_UNLOAD='1'" in s
open(T + 'colab_t4_jobs_pp2.py', 'w').write(s)
runs = [('default', 3), ('fp16b1', 3), ('fp32', 1), ('mix', 1), ('bf16', 1), ('fp32b1', 1)]
cmd = 'cd /content/t4test && (' + '; '.join(f'/content/venv/bin/python colab_t4_jobs_pp2.py {m} t4c5_{m} {n} 900' for m, n in runs) + '; echo ALL_DONE5) > run5.log 2>&1'
subprocess.Popen(['bash', '-c', cmd], start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); print('started run5')
print(subprocess.run('pgrep -af "colab_t4_jobs_p[p]" | cut -c1-100; nvidia-smi --query-gpu=memory.used --format=csv,noheader', shell=True, capture_output=True, text=True).stdout)
