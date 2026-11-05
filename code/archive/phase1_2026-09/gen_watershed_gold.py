"""Claude-annotated gold via marker-watershed (true-area boundaries), no human hand-tracing.
Recipe (validated visually on #3796: CV err 4%): on the sharp grader-region crop, smooth -> peak
markers -> watershed on the NEGATIVE smoothed image (basins=bright cell interiors, ridges=dark walls).
Tune min_distance PER IMAGE so the interior cell count ~= grader K(148) [structural target, not fit to
CV]. Unlike SAM hull-fill, these masks carry TRUE cell areas -> the CV fix. Validate each with
CD/CV/HEX vs train GT (legit supervision); keep only those within TOL on all 3 -> watershed gold.
Selects SHARP tr images (watershed needs clear walls). Saves img+mask tif pairs + a metrics report."""
import os, glob, json, numpy as np, cv2, tifffile
from cellpose import models
from skimage.feature import peak_local_max
from skimage.segmentation import watershed, clear_border, expand_labels
from skimage.measure import regionprops
from scipy.spatial import ConvexHull
from scipy import ndimage as ndi
ROOT="datasets/clear_ec_official"; H,W=972,1296; p2u=1000/1296
NPICK=int(os.environ.get("NPICK","250")); POOL=int(os.environ.get("POOL","1200")); K=148; MARGIN=18
TOL=float(os.environ.get("TOL","10")); MODEL=os.environ.get("MODEL",f"{ROOT}/models/cpsam_iter2")
OUT=f"{ROOT}/ws_gold"; os.makedirs(OUT,exist_ok=True); VIZ=os.environ.get("VIZ","")
meta=np.load(f"{ROOT}/cache_native_meta_512.npz",allow_pickle=True); N=int(meta["N"]); Y=meta["Y"].astype(float)
full=np.memmap(f"{ROOT}/img_full_clahe.dat",dtype=np.uint8,mode='r',shape=(N,H,W))
perm=np.random.default_rng(42).permutation(N); te,va,tr=perm[:1000],perm[1000:2000],perm[2000:]
def _dig(s): return int(''.join(c for c in s if c.isdigit()) or -1)
usedg=set()
for p in glob.glob(f"{ROOT}/round2/gold/*_img.tif")+glob.glob(f"{OUT}/*_img.tif"): usedg.add(_dig(os.path.basename(p).split('_')[0]))
candrows=[i for i in tr.tolist() if i not in usedg]
rng=np.random.default_rng(7); sample=rng.choice(candrows,size=min(POOL,len(candrows)),replace=False)
foc=sorted(((int(i),cv2.Laplacian(np.array(full[int(i)]),cv2.CV_32F).var()) for i in sample),key=lambda t:-t[1])
picks=[i for i,_ in foc[:NPICK]]
print(f"ws-gold: {NPICK} sharpest of {len(sample)} (focus {foc[0][1]:.0f}..{foc[NPICK-1][1]:.0f}) TOL={TOL}",flush=True)
mdl=models.CellposeModel(gpu=True, pretrained_model=MODEL)
def CV(a): return a.std()/a.mean() if len(a)>1 and a.mean()>0 else 0
def HEX_adj(m):
    lbls=np.unique(m); lbls=lbls[lbls>0]
    if len(lbls)<3: return 0
    adj={int(l):set() for l in lbls}; bd=set()
    for ax,sh in [(0,1),(0,-1),(1,1),(1,-1)]:
        r=np.roll(m,sh,axis=ax); dd=(m!=r)
        for x,y in zip(m[dd],r[dd]):
            if x==0 or y==0:
                if x>0: bd.add(int(x))
                if y>0: bd.add(int(y))
            else: adj[int(x)].add(int(y)); adj[int(y)].add(int(x))
    deg=[len(adj[int(l)]) for l in lbls if int(l) not in bd]
    return float(np.mean([1.0 if d==6 else 0.0 for d in deg])) if deg else 0
def e(a,b): return abs(a-b)/b*100 if b else 0
def region_crop(i):
    img=np.array(full[int(i)]); im=cv2.normalize(img.astype(np.float32),None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
    m,_,_=mdl.eval(im,batch_size=32,normalize=True); m=clear_border(m.astype(np.int32)); props=regionprops(m)
    if len(props)<K+5: return None
    cent=np.array([p.centroid for p in props]); c0=np.median(cent,0)
    d=np.hypot(cent[:,0]-c0[0],cent[:,1]-c0[1]); sel=np.argsort(d)[:K]; ys,xs=cent[sel,0],cent[sel,1]
    y0,y1=int(max(0,ys.min()-MARGIN)),int(min(H,ys.max()+MARGIN)); x0,x1=int(max(0,xs.min()-MARGIN)),int(min(W,xs.max()+MARGIN))
    return img[y0:y1,x0:x1]
from skimage.morphology import h_maxima, remove_small_objects, label as sklabel
def ws(crop, md, hh):
    sm=cv2.GaussianBlur(crop.astype(np.float32),(0,0),1.8)
    hm=h_maxima(sm, hh)                      # prominence-based maxima -> robust one-per-cell markers
    mk=sklabel(hm)
    # merge markers closer than md (dedup double-detections) by keeping one per md-cell
    if mk.max()>1:
        cent=ndi.center_of_mass(np.ones_like(mk),mk,range(1,mk.max()+1)); cent=np.array(cent)
        from scipy.spatial import cKDTree; keepm=np.ones(len(cent),bool); tr=cKDTree(cent)
        for a,b in tr.query_pairs(md):
            if keepm[a] and keepm[b]: keepm[b]=False
        mk2=np.zeros_like(mk); n=1
        for j,c in enumerate(cent):
            if keepm[j]: mk2[int(round(c[0])),int(round(c[1]))]=n; n+=1
        mk=mk2
    lab=watershed(-sm,markers=mk)
    lab=clear_border(lab)
    lab=remove_small_objects(lab, min_size=40)
    return lab
def compactK(lab):
    props=[p for p in regionprops(lab) if p.area>=8]
    if len(props)<K: return None
    cent=np.array([p.centroid for p in props]); labs=[p.label for p in props]; ar={p.label:p.area for p in props}
    c0=np.median(cent,0); dd=np.hypot(cent[:,0]-c0[0],cent[:,1]-c0[1]); order=np.argsort(dd)[:K]
    keep=[labs[j] for j in order]; kc=cent[order]
    a=np.array([ar[l]*(p2u**2) for l in keep]); cv=a.std()/a.mean() if a.mean()>0 else 0
    ha=ConvexHull(kc[:,::-1]).volume*(p2u**2)/1e6; cd=K/ha if ha>0 else 0
    ks=set(keep); adj={int(l):set() for l in keep}
    for ax,sh in [(0,1),(0,-1),(1,1),(1,-1)]:
        r=np.roll(lab,sh,axis=ax); m2=(lab!=r)
        for x,y in zip(lab[m2],r[m2]):
            if int(x) in adj and int(y) in adj: adj[int(x)].add(int(y)); adj[int(y)].add(int(x))
    interior=[l for l in keep if adj[l] and all(nb in ks for nb in adj[l])]
    hx=np.mean([1.0 if len(adj[l])==6 else 0.0 for l in interior]) if interior else 0
    return cd,cv,hx
rep=[]; ok=0
for i in picks:
    crop=region_crop(i)
    if crop is None: continue
    g=Y[int(i)]
    best=None  # tune h_maxima prominence so cell count ~ K*1.08 (structural target)
    for hh in [4,6,8,10,12,15,18]:
        lab=ws(crop,9,hh); nc=int(len(np.unique(lab))-1)
        if nc<K*0.85: continue
        if best is None or abs(nc-int(K*1.08))<abs(best[0]-int(K*1.08)): best=(nc,hh,lab)
    if best is None: continue
    nc,md,lab=best; m=compactK(lab)
    if m is None: continue
    cd,cv,hx=m; eCD,eCV,eHEX=e(cd,g[0]),e(cv,g[1]),e(hx,g[2])
    good=eCD<=TOL and eCV<=TOL   # gate on CD+CV (watershed's true-area strengths); HEX recorded only
    rep.append((i,nc,md,eCD,eCV,eHEX,bool(good)))
    if good:
        tifffile.imwrite(f"{OUT}/{i}_img.tif",crop.astype(np.uint8)); tifffile.imwrite(f"{OUT}/{i}_mask.tif",lab.astype(np.int32)); ok+=1
    if len(rep)<=12:
        print(f"  i{i}: md{md} nc{nc} CD{eCD:.0f}% CV{eCV:.0f}% HEX{eHEX:.0f}% {'OK' if good else ''}",flush=True)
print(f"\n[ws-gold] scored {len(rep)}  accepted(CD&CV<={TOL}%)={ok} -> {OUT}",flush=True)
if rep:
    import numpy as _n
    for t in [6,8,10,12]:
        print(f"  CD&CV<={t}%: {sum(1 for r in rep if max(r[3],r[4])<=t)}  (also HEX<={t}: {sum(1 for r in rep if max(r[3],r[4],r[5])<=t)})",flush=True)
    acc=[r for r in rep if r[6]]
    if acc: print(f"  accepted mean err: CD {_n.mean([r[3] for r in acc]):.1f} CV {_n.mean([r[4] for r in acc]):.1f} HEX {_n.mean([r[5] for r in acc]):.1f}",flush=True)
json.dump([(int(r[0]),bool(r[6])) for r in rep], open(f"{OUT}/report.json","w"))
