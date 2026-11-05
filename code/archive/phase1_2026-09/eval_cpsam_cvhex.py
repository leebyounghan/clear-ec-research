"""Eval fine-tuned cpsam on the regression val/test 512-crops: compute CV (std/mean area) and HEX
(neighbor-based) geometrically -> MAPE vs scalar GT. CV/HEX are SCALE-FREE so no pixel calibration
needed -> directly comparable to regression's CV 10.4 / HEX 10.0. If seg wins CV/HEX, hybrid
(regression CD + seg CV/HEX) beats 8.95. Uses same seed-42 split; eval only (no test tuning)."""
import os, numpy as np, json, cv2
from cellpose import models
from skimage.measure import regionprops
ROOT="datasets/clear_ec_official"; SZ=512
MODEL=os.environ.get("MODEL",f"{ROOT}/models/cpsam_pseudo25")
NVAL=int(os.environ.get("NVAL","400")); SPLIT=os.environ.get("SPLIT","va")
meta=np.load(f"{ROOT}/cache_native_meta_{SZ}.npz",allow_pickle=True); Y=meta["Y"].astype(np.float32); N=int(meta["N"])
# EXACT training preprocessing (green load()): CLAHE(2.0,8x8) then minmax-normalize to 0-255
_raw=np.memmap(f"{ROOT}/img_native_{SZ}.dat",dtype=np.uint8,mode='r',shape=(N,SZ,SZ))
_cl=cv2.createCLAHE(2.0,(8,8))
def prep(i):
    a=_cl.apply(np.ascontiguousarray(_raw[i])).astype(np.float32)
    return cv2.normalize(a,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
class _IW:
    def __getitem__(s,i): return prep(i)
imgs=_IW()
rng=np.random.default_rng(42); perm=rng.permutation(N); te,va,tr=perm[:1000],perm[1000:2000],perm[2000:]
idx={"va":va,"te":te}[SPLIT][:NVAL]
def CV(a): return 0 if (len(a)<2 or a.mean()==0) else a.std()/a.mean()
def HEX_adj(m):
    lbls=np.unique(m); lbls=lbls[lbls>0]
    if len(lbls)<3: return 0
    adj={int(l):set() for l in lbls}; border=set()
    for ax,sh in [(0,1),(0,-1),(1,1),(1,-1)]:
        r=np.roll(m,sh,axis=ax); d=(m!=r); aa=m[d]; bb=r[d]
        for x,y in zip(aa,bb):
            if x==0 or y==0:
                if x>0: border.add(int(x))
                if y>0: border.add(int(y))
            else: adj[int(x)].add(int(y)); adj[int(y)].add(int(x))
    deg=[len(adj[int(l)]) for l in lbls if int(l) not in border]
    return np.mean([1.0 if d==6 else 0.0 for d in deg]) if deg else 0
def mape(p,g): return abs(p-g)/g*100 if g else 0
mdl=models.CellposeModel(gpu=True, pretrained_model=MODEL)
print(f"eval {MODEL} on {SPLIT}[:{NVAL}]",flush=True)
CVe=[];HEXe=[];ncs=[]
for j,i in enumerate(idx):
    im=np.array(imgs[i])
    m,_,_=mdl.eval(im, batch_size=32, normalize=True)
    m=m.astype(np.int32)
    from skimage.segmentation import clear_border
    mi=clear_border(m)  # drop cells touching the crop edge (partial -> inflate CV)
    a=np.array([r.area for r in regionprops(mi)])
    CVe.append(mape(CV(a),Y[i,1])); HEXe.append(mape(HEX_adj(mi),Y[i,2])); ncs.append(int(mi.max()))
    if j<8: print(f"  i{i}: nc={int(m.max())} CVpred{CV(a):.3f}/gt{Y[i,1]:.3f} HEXpred{HEX_adj(m):.3f}/gt{Y[i,2]:.3f}",flush=True)
print(f"\n[cpsam-seg {SPLIT}] n={len(idx)} nc~{np.mean(ncs):.0f} CV_MAPE={np.mean(CVe):.2f} HEX_MAPE={np.mean(HEXe):.2f}",flush=True)
print(f"(regression baseline: CV~10.4 HEX~10.0)",flush=True)
json.dump({"split":SPLIT,"n":len(idx),"nc":float(np.mean(ncs)),"CV":float(np.mean(CVe)),"HEX":float(np.mean(HEXe))},
    open(f"{ROOT}/cpsam_seg_cvhex_{SPLIT}.json","w"),indent=1)
