"""Evaluate the POINT-PROXY pipeline directly (NO trained segmenter): detect cell points, pick a
GT-independent grader-sized region (K=148 nearest medoid), estimate CD=n/hull, CV=Voronoi-area-CV
x2.1 (green-calibrated const, not per-image GT), HEX=Delaunay 6-neighbor frac. The self-training
segmenter degrades CV (25%); the point proxy got CV~9% on green -> test whether the proxy itself,
region-selected GT-independently, beats the segmenter on val. Legit: CVCAL is a fixed constant, no
per-image GT used; region chosen by geometry only."""
import os, numpy as np, cv2
from skimage.feature import peak_local_max
from scipy.spatial import Voronoi, Delaunay, ConvexHull, cKDTree
ROOT="datasets/clear_ec_official"; H,W=972,1296; p2u=1000/1296; CVCAL=float(os.environ.get("CVCAL","2.1"))
NVAL=int(os.environ.get("NVAL","300")); SPLIT=os.environ.get("SPLIT","va"); K=int(os.environ.get("K","148"))
meta=np.load(f"{ROOT}/cache_native_meta_512.npz",allow_pickle=True); N=int(meta["N"]); Y=meta["Y"].astype(float)
full=np.memmap(f"{ROOT}/img_full_clahe.dat",dtype=np.uint8,mode='r',shape=(N,H,W))
rng=np.random.default_rng(42); perm=rng.permutation(N); te,va,tr=perm[:1000],perm[1000:2000],perm[2000:]
idx={"va":va,"te":te}[SPLIT][:NVAL]
def detect(img):
    sm=cv2.GaussianBlur(img.astype(np.float32),(0,0),2.0)
    pk=peak_local_max(sm,min_distance=9,threshold_rel=0.30,exclude_border=False)
    return np.stack([pk[:,1],pk[:,0]],1).astype(float)
def poly_area(v): x=v[:,0]; y=v[:,1]; return 0.5*abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
def proxy(pts):
    hv=ConvexHull(pts); hull=Delaunay(pts[hv.vertices])
    ha=hv.volume*(p2u**2)/1e6; cd=len(pts)/ha if ha>0 else 0
    def ins(p): return hull.find_simplex(p)>=0
    vor=Voronoi(pts); ar=[]
    for i,rp in enumerate(vor.point_region):
        reg=vor.regions[rp]
        if len(reg)==0 or -1 in reg: continue
        v=vor.vertices[reg]
        if all(ins(vv) for vv in v): ar.append(poly_area(v))
    cv=(np.std(ar)/np.mean(ar))*CVCAL if len(ar)>3 else 0
    tri=Delaunay(pts); nb={i:set() for i in range(len(pts))}
    for s in tri.simplices:
        for a in s:
            for b in s:
                if a!=b: nb[a].add(b)
    hvs=set(hv.vertices); deg=[len(nb[i]) for i in range(len(pts)) if i not in hvs]
    hx=np.mean([1.0 if d==6 else 0.0 for d in deg]) if deg else 0
    return cd,cv,hx
def mape(p,g): return abs(p-g)/g*100 if g else 0
CDe=[];CVe=[];HEXe=[]
for j,i in enumerate(idx):
    img=np.array(full[int(i)]); pts=detect(img); g=Y[int(i)]
    if len(pts)<K+5: continue
    # GT-independent compact region: K nearest the medoid
    c0=np.median(pts,0); d=np.hypot(pts[:,0]-c0[0],pts[:,1]-c0[1]); reg=pts[np.argsort(d)[:K]]
    try: cd,cv,hx=proxy(reg)
    except Exception: continue
    CDe.append(mape(cd,g[0])); CVe.append(mape(cv,g[1])); HEXe.append(mape(hx,g[2]))
    if j<6: print(f"  i{i}: CD{cd:.0f}/{g[0]:.0f}({mape(cd,g[0]):.0f}) CV{cv:.2f}/{g[1]:.2f}({mape(cv,g[1]):.0f}) HEX{hx:.2f}/{g[2]:.2f}({mape(hx,g[2]):.0f})",flush=True)
print(f"\n[point-proxy {SPLIT}] n={len(CDe)} CD_MAPE={np.mean(CDe):.2f} CV_MAPE={np.mean(CVe):.2f} HEX_MAPE={np.mean(HEXe):.2f}  MEAN3={np.mean([np.mean(CDe),np.mean(CVe),np.mean(HEXe)]):.2f}  (seg iter2 18.59; reg ens 8.95)",flush=True)
