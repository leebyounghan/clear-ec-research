"""Quantify the ROI-mismatch noise floor for CD/CV/HEX.
GT was computed on an annotator-chosen ~0.05mm^2 focused patch (per TRAINING_DATA_RELEASE.md);
we always predict from the central 512. How much do the three metrics vary across candidate
ROI-sized windows within one image? Uses cached fNLA edge maps -> full-image watershed seg ->
per-cell area + hexagon-IoU -> sliding 290x290 windows (~ROI size). Reports per metric:
  - intra-image spread (std/mean over windows)          -> noise floor if ROI unknown
  - center-window error vs GT                            -> what we effectively do now
  - best-focus-window error vs GT                        -> cheap ROI heuristic
  - oracle-window error vs GT (min over windows)         -> upper bound of ROI localization
Run on pool val rows only (~205 imgs). Honest caveat: fNLA seg errors inflate ABSOLUTE errors;
the center-vs-focus-vs-oracle DELTAS are the signal.
"""
import numpy as np, json, os
from multiprocessing import Pool
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import regionprops
from skimage.draw import polygon as skpoly
from scipy import ndimage as ndi
import cv2

ROOT="datasets/clear_ec_official"; H,W=972,1296; p2u=1000/1296; WIN=290
pm=np.load(f"{ROOT}/pool_meta.npz",allow_pickle=True); pids=[str(x) for x in pm["ids"]]; Np=int(pm["N"])
cm=np.load(f"{ROOT}/cache_native_meta_512.npz",allow_pickle=True); cids=list(cm["ids"]); Nc=int(cm["N"]); Y=cm["Y"].astype(float)
f2row={f:i for i,f in enumerate(cids)}
perm=np.random.default_rng(42).permutation(Nc); va=set(perm[1000:2000].tolist())
tasks=[(k,f2row[pids[k]]) for k in range(Np) if pids[k] in f2row and f2row[pids[k]] in va]
print(f"va pool imgs: {len(tasks)}",flush=True)

def load_mha(pid):
    with open(f"{ROOT}/train_mha/{pid}.mha","rb") as f: data=f.read()
    i=data.find(b"ElementDataFile = LOCAL\n")
    return np.frombuffer(data[i+24:],dtype=np.uint8).reshape(H,W)

def cell_hex_iou(ms,c):
    Ma=c.axis_major_length*0.8; mi=c.axis_minor_length*0.8; a=c.orientation
    hp=np.array([[Ma/2,0],[Ma/4,-mi/2],[-Ma/4,-mi/2],[-Ma/2,0],[-Ma/4,mi/2],[Ma/4,mi/2]])
    R=np.array([[np.cos(a),np.sin(a)],[-np.sin(a),np.cos(a)]]); rh=hp@R
    r0,c0,_,_=c.bbox; cy,cx=c.centroid; th=rh+np.array([np.ceil(cy-r0),np.ceil(cx-c0)])
    hm=np.zeros(ms.shape); rr,cc=skpoly(th[:,0],th[:,1],hm.shape); hm[rr,cc]=1
    inter=np.logical_and(ms,hm).sum(); uni=np.logical_or(ms,hm).sum()
    return inter/uni if uni>0 else 0.0

def one(task):
    k,row=task
    EU=np.memmap(f"{ROOT}/pool_edge_u8.dat",dtype=np.uint8,mode='r',shape=(Np,H,W))
    ep=np.asarray(EU[k]).astype(np.float32)/255.0
    interior=ndi.binary_opening(ep<0.2,iterations=1); dist=ndi.distance_transform_edt(interior)
    pk=peak_local_max(dist,min_distance=7,labels=interior,exclude_border=False)
    mk=np.zeros(ep.shape,np.int32)
    for i,(r,c) in enumerate(pk,1): mk[r,c]=i
    lab=watershed(ep,markers=ndi.label(mk>0)[0])
    cells=[]
    for c in regionprops(lab):
        if c.area<40 or c.area>6000: continue
        r0,c0,r1,c1=c.bbox
        if r0==0 or c0==0 or r1==H or c1==W: continue  # border cells: incomplete
        ms=(lab[r0:r1,c0:c1]==c.label)
        cells.append((c.centroid[0],c.centroid[1],c.area*(p2u**2),cell_hex_iou(ms,c)))
    if len(cells)<100: return None
    cy=np.array([x[0] for x in cells]); cx=np.array([x[1] for x in cells])
    ar=np.array([x[2] for x in cells]); hx=np.array([x[3] for x in cells])
    img=load_mha(pids[k]).astype(np.float32)
    gt=Y[row]
    wins=[]
    for r0 in np.linspace(0,H-WIN,6).astype(int):
        for c0 in np.linspace(0,W-WIN,8).astype(int):
            m=(cy>=r0)&(cy<r0+WIN)&(cx>=c0)&(cx<c0+WIN)
            if m.sum()<60: continue
            a=ar[m]
            cd=1e6/a.mean(); cv=a.std()/a.mean(); hh=hx[m].mean()
            foc=cv2.Laplacian(img[r0:r0+WIN,c0:c0+WIN],cv2.CV_32F).var()
            wins.append(dict(r=int(r0),c=int(c0),cd=cd,cv=cv,hex=hh,foc=foc,
                             dc=float(np.hypot(r0+WIN/2-H/2,c0+WIN/2-W/2))))
    if len(wins)<6: return None
    P=np.array([[w["cd"],w["cv"],w["hex"]] for w in wins])
    spread=(P.std(0)/P.mean(0)*100)
    err=np.abs(P-gt)/gt*100                      # (nw,3) per-window error vs GT
    ctr=err[int(np.argmin([w["dc"] for w in wins]))]
    fcs=err[int(np.argmax([w["foc"] for w in wins]))]
    orc=err.min(0); orcj=err[err.mean(1).argmin()]
    return dict(row=int(row),n=len(wins),spread=spread.tolist(),ctr=ctr.tolist(),
                fcs=fcs.tolist(),orc=orc.tolist(),orcj=orcj.tolist())

if __name__=="__main__":
    with Pool(8) as p: res=[r for r in p.map(one,tasks) if r]
    print(f"analyzed {len(res)} images",flush=True)
    def agg(key): return np.median([r[key] for r in res],0), np.mean([r[key] for r in res],0)
    names=["CD","CV","HEX"]
    for key,lab in [("spread","intra-image spread% (windows)"),("ctr","CENTER window err%"),
                    ("fcs","FOCUS-max window err%"),("orc","ORACLE per-metric err%"),("orcj","ORACLE joint-window err%")]:
        md,mn=agg(key)
        print(f"{lab:32s} median CD{md[0]:6.2f} CV{md[1]:6.2f} HEX{md[2]:6.2f} | mean CD{mn[0]:6.2f} CV{mn[1]:6.2f} HEX{mn[2]:6.2f}",flush=True)
    json.dump(res,open(f"{ROOT}/roi_region_diag.json","w"))
    print("saved -> roi_region_diag.json",flush=True)
