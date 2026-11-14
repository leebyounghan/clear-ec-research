# deploy lane 2026-11-12 (D-PKG-006): stop run3 (every job failed at ViT-L: timm.layers.test_time_pool missing from the staged runtime/timm copy,
# dropped by stage_package's test-file filter); copy the same file from the runtime's pip timm 1.0.20 if its sha256 equals the research pod's
# runtime copy (67994fbd52c8e0862aac92c64218097f13dab2e83d9edb35cba9830ffaf39c14), then rerun (run4.log).
import subprocess, hashlib, shutil, os, time
subprocess.run('pkill -f "run[3].log"; sleep 1; pkill -f "colab_t4_jobs_p[p].py"; sleep 1; pkill -f "colab_gc_wra[p].py"', shell=True)
time.sleep(3)
src = '/content/venv/lib/python3.12/site-packages/timm/layers/test_time_pool.py'
dst = '/workspace/clear-ec-research/experiments/phase2_foundation_20261002/runtime/timm/layers/test_time_pool.py'
h = hashlib.sha256(open(src, 'rb').read()).hexdigest(); print('pip copy sha', h, 'pod', '67994fbd52c8e0862aac92c64218097f13dab2e83d9edb35cba9830ffaf39c14', h == '67994fbd52c8e0862aac92c64218097f13dab2e83d9edb35cba9830ffaf39c14')
if h == '67994fbd52c8e0862aac92c64218097f13dab2e83d9edb35cba9830ffaf39c14':
    shutil.copy2(src, dst); print('copied')
    runs = [('default', 5), ('fp16b1', 5), ('fp32', 1), ('mix', 1), ('fp32b1', 1), ('bf16', 1)]
    cmd = 'cd /content/t4test && (' + '; '.join(f'/content/venv/bin/python colab_t4_jobs_pp.py {m} t4c4_{m} {n} 900' for m, n in runs) + '; echo ALL_DONE4) > run4.log 2>&1'
    subprocess.Popen(['bash', '-c', cmd], start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); print('started run4')
print(subprocess.run('pgrep -af "colab_t4_jobs_p[p]" | cut -c1-120', shell=True, capture_output=True, text=True).stdout)
