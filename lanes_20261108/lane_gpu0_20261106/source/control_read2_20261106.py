"""Control read: one-process epoch 1 against the two-rank reference, fold 0 seed 46, flips only."""
import numpy as np, json
A='experiments/lane_gpu0_20261106/arms/control_single_gpu2/f0_s46/runs/fold0/cpdino_vitb/'
R='experiments/phase2_pooling_replication_20261026/f0_s46_std/runs/fold0/cpdino_vitb/'
a=np.load(A+'predictions_epoch_01.npz',allow_pickle=True); r=np.load(R+'predictions_epoch_01.npz',allow_pickle=True)
assert np.array_equal(a['indices'],r['indices'])
y=r['target']
def m(p): return (np.abs(p-y)/y).mean(0)*100
print('control',m(a['pred']).round(4).tolist(),round(float(m(a['pred']).mean()),4))
print('reference',m(r['pred']).round(4).tolist(),round(float(m(r['pred']).mean()),4))
la,lr=np.log(a['pred']),np.log(r['pred'])
print('r log per metric',[round(float(np.corrcoef(la[:,j],lr[:,j])[0,1]),5) for j in range(3)],'median abs log diff',np.median(np.abs(la-lr),0).round(5).tolist(),'max',np.abs(la-lr).max(0).round(4).tolist())
ha=json.load(open(A+'history.json'))[0]; hr=json.load(open(R+'history.json'))[0]
for k in ('online_train_loss','gradient_norm_mean','gradient_clip_fraction','train_seconds','epoch_seconds','peak_allocated_gib'):
    print(k,ha[k],hr[k])
