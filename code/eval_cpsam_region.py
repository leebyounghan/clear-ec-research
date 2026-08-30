"""Fair GT-INDEPENDENT eval: cpsam segments the val 512-crop, pick a compact grader-sized region
(K=148 cells nearest the medoid -- densest central cluster, no GT used to place it), measure
CD/CV/HEX over that region vs scalar GT. Grader-matched (green: ~148 cells, 0.05mm^2). HEX via
dilation-closed adjacency (matches training-target estimator). K/HEX consistent with round pipeline."""
import os, numpy as np, cv2
from cellpose import models
from skimage.measure import regionprops
from skimage.segmentation import clear_border, expand_labels
from scipy.spatial import ConvexHull
ROOT="datasets/clear_ec_official"; SZ=512; p2u=1000/1296
MODEL=os.environ.get("MODEL",f"{ROOT}/models/cpsam_gold26"); NVAL=int(os.environ.get("NVAL","300")); SPLIT=os.environ.get("SPLIT","va")
K=int(os.environ.get("K","148"))
meta=np.load(f"{ROOT}/cache_native_meta_{SZ}.npz",allow_pickle=True); Y=meta["Y"].astype(np.float32); N=int(meta["N"])
raw=np.memmap(f"{ROOT}/img_native_{SZ}.dat",dtype=np.uint8,mode='r',shape=(N,SZ,SZ)); _cl=cv2.createCLAHE(2.0,(8,8))
def prep(i): return cv2.normalize(_cl.apply(np.ascontiguousarray(raw[i])).astype(np.float32),None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
rng=np.random.default_rng(42); perm=rng.permutation(N); te,va,tr=perm[:1000],perm[1000:2000],perm[2000:]
idx={"va":va,"te":te}[SPLIT][:NVAL]
def CV(a): return 0 if (len(a)<2 or a.mean()==0) else a.std()/a.mean()
def HEX_adj(m):
    exp=expand_labels(m, distance=3); lbls=np.unique(exp); lbls=lbls[lbls>0]
    if len(lbls)<3: return 0
    adj={int(l):set() for l in lbls}; border=set()
    for ax,sh in [(0,1),(0,-1),(1,1),(1,-1)]:
        r=np.roll(exp,sh,axis=ax); d=(exp!=r); aa=exp[d]; bb=r[d]
        for x,y in zip(aa,bb):
            if x==0 or y==0:
                if x>0: border.add(int(x))
                if y>0: border.add(int(y))
            else: adj[int(x)].add(int(y)); adj[int(y)].add(int(x))
    deg=[len(adj[int(l)]) for l in lbls if int(l) not in border]
    return float(np.mean([1.0 if dd==6 else 0.0 for dd in deg])) if deg else 0
def mape(p,g): return abs(p-g)/g*100 if g else 0
mdl=models.CellposeModel(gpu=True, pretrained_model=MODEL); print(f"region-eval {MODEL} on {SPLIT}[:{NVAL}] K={K}",flush=True)
CDe=[];CVe=[];HEXe=[];ncs=[]
for j,i in enumerate(idx):
    m,_,_=mdl.eval(prep(i), batch_size=32, normalize=True); m=clear_border(m.astype(np.int32))
    props=regionprops(m)
    if len(props)<K+5:
        a=np.array([p.area for p in props]); ncs.append(len(props))
        CVe.append(mape(CV(a),Y[i,1])); HEXe.append(mape(HEX_adj(m),Y[i,2]))
        try: ha=ConvexHull(np.array([p.centroid for p in props])[:,::-1]).volume*(p2u**2)/1e6; CDe.append(mape(len(props)/ha,Y[i,0]))
        except Exception: pass
        continue
    cent=np.array([p.centroid for p in props]); labs=np.array([p.label for p in props])
    c0=np.median(cent,0); d=np.hypot(cent[:,0]-c0[0],cent[:,1]-c0[1]); order=np.argsort(d)[:K]
    keep=labs[order]; kc=cent[order]
    reg=np.where(np.isin(m,keep),m,0)
    a=np.array([p.area for p in regionprops(reg)])
    ha=ConvexHull(kc[:,::-1]).volume*(p2u**2)/1e6; cd=K/ha
    CDe.append(mape(cd,Y[i,0])); CVe.append(mape(CV(a),Y[i,1])); HEXe.append(mape(HEX_adj(reg),Y[i,2])); ncs.append(int(len(keep)))
    if j<6: print(f"  i{i}: nc={len(keep)} CD{cd:.0f}/{Y[i,0]:.0f}({mape(cd,Y[i,0]):.0f}) CVe{mape(CV(a),Y[i,1]):.0f} HEXe{mape(HEX_adj(reg),Y[i,2]):.0f}",flush=True)
print(f"\n[region-eval {SPLIT}] n={len(idx)} nc~{np.mean(ncs):.0f} CD_MAPE={np.mean(CDe):.2f} CV_MAPE={np.mean(CVe):.2f} HEX_MAPE={np.mean(HEXe):.2f}  MEAN3={np.mean([np.mean(CDe),np.mean(CVe),np.mean(HEXe)]):.2f}  (reg ens 8.95)",flush=True)
