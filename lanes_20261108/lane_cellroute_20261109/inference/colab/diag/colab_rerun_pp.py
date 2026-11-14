# deploy lane 2026-11-12 (D-PKG-006): rerun the real-T4 jobs with the fixed image environment (Dockerfile.freeze ENV now sets PYTHONPATH = the
# runtime folder and PYTORCH_CUDA_ALLOC_CONF = expandable_segments:True, as on the research pod). Derives colab_t4_jobs_pp.py from the uploaded
# driver on the runtime (only the job environment changes) and starts the jobs in the background; log /content/t4test/run2.log.
import subprocess
T = '/content/t4test/'
s = open(T + 'colab_t4_jobs.py').read()
s = s.replace("k not in ('PYTHONPATH',)", "k not in ()")
s = s.replace("base.update(PYTHONUNBUFFERED='1',", "base.update(PYTHONPATH='/workspace/clear-ec-research/experiments/phase2_foundation_20261002/runtime', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', PYTHONUNBUFFERED='1',")
assert "PYTORCH_CUDA_ALLOC_CONF='expandable" in s and "k not in ()" in s
open(T + 'colab_t4_jobs_pp.py', 'w').write(s)
runs = [('default', 5), ('fp16b1', 5), ('fp32', 1), ('mix', 1), ('fp32b1', 1), ('bf16', 1)]
cmd = 'cd /content/t4test && (' + '; '.join(f'/content/venv/bin/python colab_t4_jobs_pp.py {m} t4c2_{m} {n} 900' for m, n in runs) + '; echo ALL_DONE2) > run2.log 2>&1'
subprocess.Popen(['bash', '-c', cmd], start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print('started', cmd)
