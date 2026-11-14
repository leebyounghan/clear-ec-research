# deploy lane 2026-11-12 (D-PKG-006): after run5's default + fp16b1 jobs, stop run5's remaining one-frame jobs and run the tiers (the t1 archive
# holds every model file of p / t2 / t3) with fp16 + one detector copy, then t1 fp32 / mix / native bf16 one frame each (600 s cap). run6.log.
import subprocess, time
subprocess.run('pkill -f "run[5].log"; sleep 1; pkill -f "colab_t4_jobs_p[p]2.py"; sleep 1; pkill -f "colab_gc_wra[p].py"', shell=True); time.sleep(5)
T = '/content/t4test/'; s = open(T + 'colab_t4_jobs_pp2.py').read()
s = s.replace("CLEAR_EC_UNLOAD='1',", "CLEAR_EC_UNLOAD='1', **({'CLEAR_EC_TIER': os.environ['TIERX']} if os.environ.get('TIERX') else {}),")
assert "TIERX" in s
open(T + 'colab_t4_jobs_pp3.py', 'w').write(s)
runs = [('t3', 'fp16b1', 2, 900), ('t2', 'fp16b1', 2, 900), ('p', 'fp16b1', 2, 900), ('', 'fp32', 1, 600), ('', 'mix', 1, 600), ('', 'bf16', 1, 600)]
cmd = 'cd /content/t4test && (' + '; '.join(f'TIERX={t} /content/venv/bin/python colab_t4_jobs_pp3.py {m} t4c6_{t or "t1"}_{m} {n} {to}' for t, m, n, to in runs) + '; echo ALL_DONE6) > run6.log 2>&1'
subprocess.Popen(['bash', '-c', cmd], start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); print('started run6')
print(subprocess.run('pgrep -af "colab_t4_jobs_p[p]" | cut -c1-100', shell=True, capture_output=True, text=True).stdout)
