import os, shutil, signal, subprocess, time
from pathlib import Path
L = Path('/workspace/clear-ec-research/experiments/lane_gpu1_20261106')
src = L/'source/run_e2e_v2_20261107.sh'
st = src.stat(); print('v2 mtime', time.strftime('%FT%TZ', time.gmtime(st.st_mtime)), 'size', st.st_size, 'inode', st.st_ino)
print('bash -n', subprocess.run(['bash','-n',str(src)],capture_output=True,text=True))
# A private, read-only copy: nothing else writes to it while it runs.
new = L/'source/run_e2e_gpu1_20261107.sh'
shutil.copyfile(src, new); os.chmod(new, 0o555)
q = (L/'queue.tsv').read_text().splitlines()
names = [r.split('\t')[0] for r in q]
i = names.index('e2e_cropopt_both_f4')
q.insert(i+1, f"e2e_cropopt_both_f4_r\tbash {new} cropopt_both_f4 4 46")
# fold 1 SSL has been running two minutes and has no checkpoint: it goes back behind the end-to-end run under a new queue name, same arm.
j = [r.split('\t')[0] for r in q].index('cropopt_both_f1')
f1 = q[j]; q.insert(j+1, 'cropopt_both_f1_r\t' + f1.split('\t',1)[1])
t = L/'queue.tmp'; t.write_text('\n'.join(q)+'\n'); t.replace(L/'queue.tsv')
print([r.split('\t')[0] for r in q][5:])
for r in subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines():
    if 'ssl_views_20261106.py cropopt_both_f1' in r and 'bash -c' not in r:
        os.kill(int(r.split()[0]), signal.SIGTERM)
time.sleep(25)
print(open(L/'done.tsv').read()[-200:])
print(subprocess.check_output(f"nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader -i 1; nvidia-smi -i 1 --query-compute-apps=pid,used_memory --format=csv,noheader; tail -4 {L}/logs/e2e_cropopt_both_f4_r.log | cut -c1-200; ls {L}/e2e/cropopt_both_f4/f4_s46", shell=True, text=True))
