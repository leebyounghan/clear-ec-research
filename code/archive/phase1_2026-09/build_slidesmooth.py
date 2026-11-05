"""Build slide-averaged target arrays (label denoising; adopted improvement, -0.20 test OV in A/B).
Frames sharing a slide_id have very similar GT (intra-slide rel-std CD 2.5 / CV 5.3 / HEX 5.5 %); averaging
their labels reduces per-frame label noise. Two outputs:
  Y_slidesmooth.npy      -> averages over TRAIN-split frames only (seed-42 split) => safe for held-out A/B
  Y_slidesmooth9000.npy  -> averages over ALL frames             => for the all-9000 final retrain
Both are aligned to cache_native_meta_512.npz row order. Run after cache_native.py."""
import pandas as pd, numpy as np
from collections import defaultdict
ROOT="datasets/clear_ec_official"
df=pd.read_csv(f"{ROOT}/final_train_ids.csv"); df["stem"]=df["ID"].astype(str).str.replace(".bmp","",regex=False)
cm=np.load(f"{ROOT}/cache_native_meta_512.npz",allow_pickle=True); cids=[str(x) for x in cm["ids"]]; N=int(cm["N"]); Y=cm["Y"].astype(np.float64)
s2r={s:i for i,s in enumerate(cids)}
perm=np.random.default_rng(42).permutation(N); tr=set(perm[2000:].tolist())
groups=defaultdict(list)
for _,r in df.iterrows():
    i=s2r.get(r["stem"])
    if i is not None: groups[r["slide_id"]].append(i)

def build(restrict_train):
    Ys=Y.copy(); nsm=0
    for s,rows in groups.items():
        use=[i for i in rows if (i in tr)] if restrict_train else rows
        if len(use)<2: continue
        m=Y[use].mean(0)
        for i in use: Ys[i]=m
        nsm+=len(use)
    return Ys.astype(np.float32), nsm

Ytr,n1=build(True);  np.save(f"{ROOT}/Y_slidesmooth.npy",Ytr)
Yall,n2=build(False); np.save(f"{ROOT}/Y_slidesmooth9000.npy",Yall)
print(f"Y_slidesmooth.npy: smoothed {n1} train rows | Y_slidesmooth9000.npy: smoothed {n2} rows (all)")
