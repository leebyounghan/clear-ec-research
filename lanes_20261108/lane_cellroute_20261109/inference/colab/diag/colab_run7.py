# deploy lane 2026-11-12 (D-PKG-006): tiers t2 / t3 on the real T4 in recipe mode. The Colab code tree is the t1 context, which has no
# lane_cv deploy_t2 / deploy_t3 folder (each tier's own context has its folder). deploy_t2 is byte-identical to deploy_t1 (sha256 checked on the
# pod: cv_deploy_params.json ad9cec5d40371d8c, cv_ridge_params.npz 744b16d26998e765) -> exact. deploy_t3 differs only in one CV stacking weight
# (fz:autocorr 0.125 vs 0.25; same components, same compute): t1's files there make the t3 job's wall valid but its CV output TIMING-ONLY.
import subprocess, hashlib, os, shutil
W = '/workspace/clear-ec-research/experiments/lane_cv_20261108/'
sh = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]
assert sh(W + 'deploy_t1/cv_deploy_params.json') == 'ad9cec5d40371d8c' and sh(W + 'deploy_t1/cv_ridge_params.npz') == '744b16d26998e765'
for t in ('deploy_t2', 'deploy_t3'):
    os.makedirs(W + t, exist_ok=True)
    for f in ('cv_deploy_params.json', 'cv_ridge_params.npz'): shutil.copy2(W + 'deploy_t1/' + f, W + t + '/' + f)
open(W + 'deploy_t3/COLAB_TIMING_ONLY.txt', 'w').write('t1 CV params copied here for a timing run; not the t3 CV stack (fz:autocorr weight differs)\n')
print('pending run6:', subprocess.run('pgrep -f "colab_t4_jobs_p[p]3.py"', shell=True, capture_output=True, text=True).stdout.strip() or 'none')
runs = [('t2', 'fp16b1', 2), ('t3', 'fp16b1', 2)]
subprocess.run('pkill -f "run[7].log"', shell=True)      # a first queued copy waited on its own command line
cmd = 'cd /content/t4test && until grep -q ALL_DONE6 run6.log; do sleep 20; done; (' + '; '.join(f'TIERX={t} /content/venv/bin/python colab_t4_jobs_pp3.py {m} t4c7_{t}_{m} {n} 900' for t, m, n in runs) + '; echo ALL_DONE7) > run7.log 2>&1'
subprocess.Popen(['bash', '-c', cmd], start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); print('queued run7 (starts after run6)')
