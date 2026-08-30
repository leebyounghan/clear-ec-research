"""ROUND-r candidate generation, SELF-TRAINING source (cpsam's OWN dense output).
For each tr-batch image: cpsam segments the full CLAHE image -> dense instance mask. Then search
grader-sized candidate ROIs over position x size (green-grounded: K in 110..169, hull area in
0.038..0.066 mm^2), measure CD/CV/HEX from the MASK (HEX via dilation-closed adjacency), pick the
lowest-error ROI vs train GT. Saves per-image best mask (region-restricted) + metrics json.
Legit: train GT selects WHICH region to keep (allowed supervision); eval later uses GT-independent region."""
import os, glob, json, numpy as np, cv2, tifffile
from cellpose import models
from skimage.measure import regionprops
from skimage.segmentation import clear_border, expand_labels
from scipy.spatial import cKDTree, ConvexHull
ROOT="datasets/clear_ec_official"; H,W=972,1296; p2u=1000/1296
MODEL=os.environ.get("MODEL",f"{ROOT}/models/cpsam_gold26"); RD=os.environ.get("RD",f"{ROOT}/round1")
LIMIT=int(os.environ.get("LIMIT","0"))
K_RANGE=[110,130,148,169]; AREA_MM2=(0.038,0.066); STRIDE=60
meta=np.load(f"{ROOT}/cache_native_meta_512.npz",allow_pickle=True); N=int(meta["N"]); Y=meta["Y"].astype(float)
full=np.memmap(f"{ROOT}/img_full_clahe.dat",dtype=np.uint8,mode='r',shape=(N,H,W))
batch=json.load(open(f"{RD}/batch.json"));
if LIMIT: batch=batch[:LIMIT]
OUT=f"{RD}/cpsam"; os.makedirs(OUT,exist_ok=True)
def e(a,b): return abs(a-b)/b if b else 9
mdl=models.CellposeModel(gpu=True, pretrained_model=MODEL)
print(f"[cpsam-cand] model={MODEL} imgs={len(batch)} K={K_RANGE} area={AREA_MM2}",flush=True)
res={}
for n,i in enumerate(batch):
    img=np.array(full[int(i)]); g=Y[int(i)]
    im=cv2.normalize(img.astype(np.float32),None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
    m,_,_=mdl.eval(im, batch_size=32, normalize=True); m=clear_border(m.astype(np.int32))
    props=regionprops(m)
    if len(props)<max(K_RANGE)+5:
        res[str(i)]={"ok":False,"reason":f"few_cells_{len(props)}"}
        if n<8: print(f"  i{i}: few cells {len(props)}",flush=True)
        continue
    labs=np.array([p.label for p in props]); cent=np.array([p.centroid for p in props])  # (row=y,col=x)
    area_px={p.label:p.area for p in props}
    # global adjacency + border via dilation-closed labels (HEX fix)
    exp=expand_labels(m, distance=3)
    adj={int(l):set() for l in labs}; gborder=set()
    for ax,sh in [(0,1),(0,-1),(1,1),(1,-1)]:
        r=np.roll(exp,sh,axis=ax); d=(exp!=r); aa=exp[d]; bb=r[d]
        for x,y in zip(aa,bb):
            if x==0 or y==0:
                if x>0: gborder.add(int(x))
                if y>0: gborder.add(int(y))
            elif int(x) in adj and int(y) in adj: adj[int(x)].add(int(y)); adj[int(y)].add(int(x))
    tree=cKDTree(cent)
    best=None
    ys,xs=cent[:,0],cent[:,1]
    for cy in range(int(ys.min())+30,int(ys.max())-30,STRIDE):
        for cx in range(int(xs.min())+30,int(xs.max())-30,STRIDE):
            for K in K_RANGE:
                if K>len(labs): continue
                dd,idx=tree.query([cy,cx],k=K); rl=labs[idx]; rc=cent[idx]
                try: ha=ConvexHull(rc[:,::-1]).volume*(p2u**2)/1e6
                except Exception: continue
                if not (AREA_MM2[0]<=ha<=AREA_MM2[1]): continue
                cd=K/ha
                ar=np.array([area_px[int(l)] for l in rl],float); cv=ar.std()/ar.mean() if ar.mean()>0 else 0
                rs=set(int(l) for l in rl)
                interior=[l for l in rs if l not in gborder and adj[l].issubset(rs)]
                hexf=np.mean([1.0 if len(adj[l])==6 else 0.0 for l in interior]) if interior else 0
                err=e(cd,g[0])+e(cv,g[1])+e(hexf,g[2])
                if best is None or err<best[0]: best=(err,rl.copy(),cd,cv,hexf)
    if best is None:
        res[str(i)]={"ok":False,"reason":"no_valid_region"}
        if n<8: print(f"  i{i}: no valid region",flush=True)
        continue
    err,rl,cd,cv,hexf=best
    reg=np.where(np.isin(m,rl),m,0).astype(np.int32)
    tifffile.imwrite(f"{OUT}/{i}_mask.tif",reg)
    res[str(i)]={"ok":True,"cd":round(cd,1),"cv":round(cv,4),"hex":round(hexf,4),
                 "eCD":round(e(cd,g[0])*100,1),"eCV":round(e(cv,g[1])*100,1),"eHEX":round(e(hexf,g[2])*100,1),
                 "err":round(err,4),"nreg":int(len(rl))}
    if n<8 or n%50==0:
        print(f"  i{i}: CD {g[0]:.0f}->{cd:.0f}({e(cd,g[0])*100:.0f}%) CV {g[1]:.2f}->{cv:.2f}({e(cv,g[1])*100:.0f}%) HEX {g[2]:.2f}->{hexf:.2f}({e(hexf,g[2])*100:.0f}%)",flush=True)
json.dump(res,open(f"{RD}/cpsam_cand.json","w"))
ok=[v for v in res.values() if v.get("ok")]
print(f"\n[cpsam-cand DONE] {len(ok)}/{len(res)} produced a region",flush=True)
if ok:
    for tol in [8,10,12,15]:
        g=[v for v in ok if v['eCD']<=tol and v['eCV']<=tol and v['eHEX']<=tol]
        print(f"  all3<={tol}%: {len(g)} imgs",flush=True)
