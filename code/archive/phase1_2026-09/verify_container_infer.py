"""Verify the container's inference logic (submission/inference.py) reproduces the expected s2
numbers on val, using the staged local weights. Feeds img_native_512 (= central 512 crop, exactly
what the container derives from the full .mha) into the same predict() and reports CD/CV/HEX MAPE.
Expected ~ CD6.4 CV10.4 HEX10.1 (clahe_s2). Confirms the log-space consts + preprocessing match."""
import os, sys, numpy as np, torch
sys.path.insert(0,"submission")
import inference as INF
INF.MODEL_PATH=__import__("pathlib").Path("submission_model")
ROOT="datasets/clear_ec_official"
meta=np.load(f"{ROOT}/cache_native_meta_512.npz",allow_pickle=True); Y=meta["Y"].astype(np.float32); N=int(meta["N"])
imgs=np.array(np.memmap(f"{ROOT}/img_native_512.dat",dtype=np.uint8,mode='r',shape=(N,512,512)))
rng=np.random.default_rng(42); perm=rng.permutation(N); va=perm[1000:2000]
idx=va[:int(os.environ.get("NVAL","120"))]
es=[]
for j,i in enumerate(idx):
    cd,cv,hx=INF.predict(imgs[int(i)])   # img already central-512
    g=Y[int(i)]; es.append([abs(cd-g[0])/g[0]*100,abs(cv-g[1])/g[1]*100,abs(hx-g[2])/g[2]*100])
    if j<4: print(f"  i{i}: CD{cd:.0f}/{g[0]:.0f} CV{cv:.3f}/{g[1]:.3f} HEX{hx:.3f}/{g[2]:.3f}",flush=True)
e=np.array(es).mean(0)
print(f"\n[verify] n={len(idx)} CD {e[0]:.2f} CV {e[1]:.2f} HEX {e[2]:.2f} OV {e.mean():.2f}  (expect ~s2: 6.4/10.4/10.1)",flush=True)
