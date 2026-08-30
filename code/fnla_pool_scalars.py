"""Compute fNLA-seg predicted scalars for all pool images (for the regression+fNLA stacking fusion).
Cached pool_edge -> segment (interior=ep<0.2, dist-peaks, watershed) -> compact K=148 region ->
CD/CV/HEX(hexagon-IoU, the CHALLENGE definition) + aux features (n_cells, focus). Saves per-row.
Split-tagged (tr/va/te) so a meta-learner trains on tr, evals on val/test. CPU-heavy; run in bg."""
import numpy as np, json, cv2
from skimage.feature import peak_local_max
from skimage.segmentation import watershed, clear_border
from skimage.measure import regionprops
from skimage.draw import polygon as skpoly
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull
ROOT="datasets/clear_ec_official"; H,W=972,1296; p2u=1000/1296; K=148
pm=np.load(f"{ROOT}/pool_meta.npz",allow_pickle=True); pids=pm["ids"]; Np=int(pm["N"])
EU=np.memmap(f"{ROOT}/pool_edge_u8.dat",dtype=np.uint8,mode='r',shape=(Np,H,W))
cm=np.load(f"{ROOT}/cache_native_meta_512.npz",allow_pickle=True); cids=list(cm["ids"]); Nc=int(cm["N"]); Y=cm["Y"].astype(float)
f2row={f:i for i,f in enumerate(cids)}
full=np.memmap(f"{ROOT}/img_full_clahe.dat",dtype=np.uint8,mode='r',shape=(Nc,H,W))
perm=np.random.default_rng(42).permutation(Nc); te,va=set(perm[:1000].tolist()),set(perm[1000:2000].tolist())
def segment(ep,t=0.2,md=7):
    interior=ep<t; interior=ndi.binary_opening(interior,iterations=1); dist=ndi.distance_transform_edt(interior)
    pk=peak_local_max(dist,min_distance=md,labels=interior,exclude_border=False)
    mk=np.zeros(ep.shape,np.int32)
    for i,(r,c) in enumerate(pk,1): mk[r,c]=i
    mk=ndi.label(mk>0)[0]; return watershed(ep,markers=mk)
def hexIoU(m,keep):
    hs=[]
    for c in regionprops(m):
        if c.label not in keep: continue
        r0,c0,r1,c1=c.bbox; ms=(m[r0:r1,c0:c1]==c.label)
        Ma=c.axis_major_length*0.8; mi=c.axis_minor_length*0.8; a=c.orientation
        hp=np.array([[Ma/2,0],[Ma/4,-mi/2],[-Ma/4,-mi/2],[-Ma/2,0],[-Ma/4,mi/2],[Ma/4,mi/2]])
        R=np.array([[np.cos(a),np.sin(a)],[-np.sin(a),np.cos(a)]]); rh=hp@R
        cy,cx=c.centroid; th=rh+np.array([np.ceil(cy-r0),np.ceil(cx-c0)])
        hm=np.zeros(ms.shape); rr,cc=skpoly(th[:,0],th[:,1],hm.shape); hm[rr,cc]=1
        inter=np.logical_and(ms,hm).sum(); uni=np.logical_or(ms,hm).sum()
        hs.append(inter/uni if uni>0 else 0)
    return float(np.mean(hs)) if hs else 0
out={}
for k in range(Np):
    row=f2row.get(str(pids[k]))
    if row is None: continue
    sp="te" if row in te else ("va" if row in va else "tr")
    lab=clear_border(segment(np.asarray(EU[k]).astype(np.float32)/255.0))
    props=[p for p in regionprops(lab) if p.area>=40]
    if len(props)<K: continue
    cent=np.array([p.centroid for p in props]); labs=[p.label for p in props]; ar={p.label:p.area for p in props}
    c0=np.median(cent,0); d=np.hypot(cent[:,0]-c0[0],cent[:,1]-c0[1]); order=np.argsort(d)[:K]
    keep=set(labs[j] for j in order); kc=cent[order]
    a=np.array([ar[l]*(p2u**2) for l in keep]); cv=float(a.std()/a.mean())
    ha=ConvexHull(kc[:,::-1]).volume*(p2u**2)/1e6; cd=float(K/ha)
    hx=hexIoU(lab,keep)
    foc=float(cv2.Laplacian(np.asarray(full[row]),cv2.CV_32F).var())
    out[str(row)]={"sp":sp,"fnla":[cd,cv,hx],"ncell":len(props),"focus":foc,"gt":Y[row].tolist()}
    if len(out)%300==0: print(f"{len(out)} done",flush=True)
json.dump(out,open(f"{ROOT}/fnla_pool_scalars.json","w"))
n=len(out); sps={s:sum(1 for v in out.values() if v['sp']==s) for s in ['tr','va','te']}
print(f"DONE {n} rows  tr{sps['tr']} va{sps['va']} te{sps['te']} -> fnla_pool_scalars.json",flush=True)
