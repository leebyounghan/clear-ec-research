"""Test marker-controlled watershed as a CV source. Endothelium: bright cell interiors, dark borders.
Seed watershed with detected point-markers, flood over the gradient so basins snap to the real dark
borders -> cell areas reflect true size heterogeneity (unlike SAM hull-fill which smooths -> low CV).
Eval GT-independently on val: detect pts -> medoid K region -> marker-watershed on the region ->
CV/HEX/CD from watershed cells vs GT. If CV beats the segmenter's 25%, watershed gold is the fix."""
import os, numpy as np, cv2
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import regionprops
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull, Delaunay, cKDTree
ROOT="datasets/clear_ec_official"; H,W=972,1296; p2u=1000/1296
NVAL=int(os.environ.get("NVAL","300")); SPLIT=os.environ.get("SPLIT","va"); K=int(os.environ.get("K","148"))
meta=np.load(f"{ROOT}/cache_native_meta_512.npz",allow_pickle=True); N=int(meta["N"]); Y=meta["Y"].astype(float)
full=np.memmap(f"{ROOT}/img_full_clahe.dat",dtype=np.uint8,mode='r',shape=(N,H,W))
rng=np.random.default_rng(42); perm=rng.permutation(N); te,va,tr=perm[:1000],perm[1000:2000],perm[2000:]
idx={"va":va,"te":te}[SPLIT][:NVAL]
def detect(img):
    sm=cv2.GaussianBlur(img.astype(np.float32),(0,0),2.0)
    pk=peak_local_max(sm,min_distance=9,threshold_rel=0.30,exclude_border=False)
    return pk  # (row,col)
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
    return float(np.mean([1.0 if dd==6 else 0.0 for dd in deg])) if deg else 0
def mape(p,g): return abs(p-g)/g*100 if g else 0
CDe=[];CVe=[];HEXe=[]
for j,i in enumerate(idx):
    img=np.array(full[int(i)]); pk=detect(img); g=Y[int(i)]
    if len(pk)<K+5: continue
    c0=np.median(pk,0); d=np.hypot(pk[:,0]-c0[0],pk[:,1]-c0[1]); reg=pk[np.argsort(d)[:K]]
    # crop bbox around region + margin
    y0,y1=reg[:,0].min()-15,reg[:,0].max()+15; x0,x1=reg[:,1].min()-15,reg[:,1].max()+15
    y0,x0=max(0,y0),max(0,x0); y1,x1=min(H,y1),min(W,x1)
    sub=img[y0:y1,x0:x1].astype(np.float32)
    mk=np.zeros(sub.shape,np.int32)
    for n,(ry,rx) in enumerate(reg,1): mk[int(ry-y0),int(rx-x0)]=n
    grad=ndi.gaussian_gradient_magnitude(sub,1.0)  # ridges at dark borders
    lab=watershed(grad, markers=mk)
    # restrict to region hull
    hull=Delaunay(reg[:,::-1][ConvexHull(reg[:,::-1]).vertices]); yy,xx=np.mgrid[y0:y1,x0:x1]
    inh=hull.find_simplex(np.stack([xx.ravel(),yy.ravel()],1)).reshape(sub.shape)>=0
    lab[~inh]=0
    props=regionprops(lab); a=np.array([p.area for p in props])*(p2u**2)
    if len(a)<10: continue
    ha=ConvexHull(reg[:,::-1]).volume*(p2u**2)/1e6; cd=len(a)/ha
    CDe.append(mape(cd,g[0])); CVe.append(mape(CV(a),g[1])); HEXe.append(mape(HEX_adj(lab),g[2]))
    if j<6: print(f"  i{i}: nc={len(a)} CD{cd:.0f}/{g[0]:.0f}({mape(cd,g[0]):.0f}) CV{CV(a):.2f}/{g[1]:.2f}({mape(CV(a),g[1]):.0f}) HEX{HEX_adj(lab):.2f}/{g[2]:.2f}({mape(HEX_adj(lab),g[2]):.0f})",flush=True)
print(f"\n[watershed {SPLIT}] n={len(CDe)} CD_MAPE={np.mean(CDe):.2f} CV_MAPE={np.mean(CVe):.2f} HEX_MAPE={np.mean(HEXe):.2f}  MEAN3={np.mean([np.mean(CDe),np.mean(CVe),np.mean(HEXe)]):.2f}  (seg iter2 18.59; reg 8.95)",flush=True)
