#!/bin/bash
L=/workspace/clear-ec-research/experiments/lane_gpu1_20261106; PY=/workspace/venv/bin/python; R=experiments/lane_gpu1_20261106
n=cropxfer_both_f4
printf '%s\t%s\n' "$n" "$PY -u $R/source/ssl_views_20261106.py $n --fold 4 --crop 544 --micro 16 --optics transfer --pool both && $PY -u $R/source/frozen_features_20261106.py $n $R/ssl/$n/run/last.pt && CUDA_VISIBLE_DEVICES= $PY -u $R/source/frozen_read_20261106.py $n 4 --ref ssl5_f4" > $L/q.tmp
# Goes before the long 20-epoch arm in the short stream.
grep -vP '^cropopt20_both_f4\t' $L/queue_short.tsv >> $L/q.tmp
grep -P '^cropopt20_both_f4\t' $L/queue_short.tsv >> $L/q.tmp
mv $L/q.tmp $L/queue_short.tsv
cut -c1-40 $L/queue_short.tsv
cat $L/done_short.tsv 2>/dev/null
cat /sys/fs/cgroup/cpu.stat | head -3; cat /sys/fs/cgroup/memory.current
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv -i 1
tail -1 $L/logs/cropopt_both_f0.log | cut -c1-110; tail -1 $L/logs/cropopt_both_s1004_f4.log | cut -c1-110
cd /workspace/clear-ec-research; CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 PYTHONPATH=$L/source:/workspace/clear-ec-research/experiments/phase2_foundation_20261002/runtime $PY - <<'PY'
import numpy as np, sys
sys.path.insert(0,'experiments/lane_gpu1_20261106/source')
import ssl_views_20261106 as V, time
raw=np.memmap(V.RAW,mode='r',dtype=np.uint8,shape=(V.N,V.H,V.W))
st=V.stats(); rng=np.random.default_rng(0); t=time.time()
for i in (100,5000):
    so=np.where(st['src']==st['src'][i])[0]; o=V.transfer(np.asarray(raw[i]),i,int(rng.choice(so)),int(st['src'][i]),rng); print(o.shape,o.dtype,round(float(o.mean()),1))
print('sec per view',round((time.time()-t)/2,3))
class A: pass
a=A(); a.optics='transfer'; a.views=2; a.seed=1003; a.crop=544
ds=V.Views(np.arange(50),0,a); x,ix=ds[3]; print(x.shape,ix,float(x.min()),float(x.max()))
PY
