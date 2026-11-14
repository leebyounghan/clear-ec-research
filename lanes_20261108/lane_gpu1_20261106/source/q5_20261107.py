import os, signal, subprocess, time
from pathlib import Path
import numpy as np
L = Path('/workspace/clear-ec-research/experiments/lane_gpu1_20261106'); PY='/workspace/venv/bin/python'; R='experiments/lane_gpu1_20261106'
for f,p in {1:'ssl_screen_20260911/preflight/fold1.npz',2:'overnight_target_20260912/preflight/fold2.npz',3:'overnight_target_20260912/preflight/fold3.npz'}.items():
    with np.load('/workspace/clear-ec-research/experiments/'+p) as z: print('fold',f,'train',len(z['train']),'val',len(z['validation']),'train%32',len(z['train'])%32)
print('all 8954 %32',8954%32)
# Stop the short stream's job: this lane's seed-1004 SSL, about 8 minutes in. Requeued below under a new name, because a partial run folder refuses to resume.
out = subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines()
for r in out:
    if 'ssl_views_20261106.py cropopt_both_s1004_f4' in r and 'bash -c' not in r:
        pid=int(r.split()[0]); os.kill(pid,signal.SIGTERM); print('sent TERM to',r[:110])
time.sleep(4)
RECIPE='--crop 544 --micro 16 --optics measured --pool both'
def arm(n,f,flags):
    return f"{n}\t{PY} -u {R}/source/ssl_views_20261106.py {n} --fold {f} {flags} && {PY} -u {R}/source/frozen_features_20261106.py {n} {R}/ssl/{n}/run/last.pt && CUDA_VISIBLE_DEVICES= {PY} -u {R}/source/frozen_read_20261106.py {n} {f} --ref ssl5_f{f}"
q=(L/'queue.tsv').read_text().splitlines()
keep=[r for r in q if r.split('\t')[0] in ('crop_mean_f4','crop_std_f4','cropopt_mean_f4','cropopt_std_f4','cropopt_both_f4','cropopt_both_f0')]
mean_f0=[r for r in q if r.split('\t')[0]=='cropopt_mean_f0']
assert len(keep)==6 and len(mean_f0)==1
new=keep+[f"e2e_cropopt_both_f4\tbash {L}/source/run_e2e.sh cropopt_both_f4 4 46"]
new+=[f"cropopt_both_{'all' if f==99 else 'f%d'%f}\t{PY} -u {R}/source/ssl_views_20261106.py cropopt_both_{'all' if f==99 else 'f%d'%f} --fold {f} {RECIPE}" for f in (1,3,2,99)]
new+=mean_f0
new.append(arm('cropopt_both_seed1004_f4',4,RECIPE+' --seed 1004'))
new.append(arm('cropxfer_both_f4',4,'--crop 544 --micro 16 --optics transfer --pool both'))
new.append(arm('cropoptlr3_both_f4',4,RECIPE+' --encoder-lr 3e-5'))
new.append(arm('cropopt20_both_f4',4,RECIPE+' --epochs 20'))
t=L/'queue.tmp'; t.write_text('\n'.join(new)+'\n'); t.replace(L/'queue.tsv')
t=L/'queue_short.tmp'; t.write_text(''); t.replace(L/'queue_short.tsv')
print('\n'.join(r.split('\t')[0] for r in new))
