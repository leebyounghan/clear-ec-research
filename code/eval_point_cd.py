"""Evaluate the point-detector's CD on val (hybrid plan: CD from points, CV/HEX from regression).
pointdet (cpsam enc + heatmap dec) on val 512-crop -> peaks -> points. CD = n/hull_area two ways:
(a) ALL points in the 512 crop, (b) compact K=148 nearest medoid (grader-matched). MAPE vs GT CD.
Compare to regression CD 6.5%. No val tuning of the detector (it was trained only on 25 green)."""
import os, numpy as np, torch, torch.nn as nn, cv2
from skimage.feature import peak_local_max
from scipy.spatial import ConvexHull
from cellpose import models
ROOT="datasets/clear_ec_official"; SZ=512; dev="cuda"; p2u=1000/1296
MIND=int(os.environ.get("MIND","7")); THR=float(os.environ.get("THR","0.25")); NVAL=int(os.environ.get("NVAL","400")); SPLIT=os.environ.get("SPLIT","va")
meta=np.load(f"{ROOT}/cache_native_meta_{SZ}.npz",allow_pickle=True); N=int(meta["N"]); Y=meta["Y"].astype(float)
raw=np.memmap(f"{ROOT}/img_native_{SZ}.dat",dtype=np.uint8,mode='r',shape=(N,SZ,SZ)); _cl=cv2.createCLAHE(2.0,(8,8))
def prep(i):
    a=_cl.apply(np.ascontiguousarray(raw[i])).astype(np.float32); a=cv2.normalize(a,None,0,255,cv2.NORM_MINMAX)
    lo,hi=np.percentile(a,(1,99)); return np.clip((a-lo)/(hi-lo+1e-6),0,1).astype(np.float32)
rng=np.random.default_rng(42); perm=rng.permutation(N); te,va,tr=perm[:1000],perm[1000:2000],perm[2000:]
idx={"va":va,"te":te}[SPLIT][:NVAL]
class Det(nn.Module):
    def __init__(s,enc):
        super().__init__(); s.enc=enc
        def up(i,o): return nn.Sequential(nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),nn.Conv2d(i,o,3,1,1),nn.GroupNorm(8,o),nn.GELU())
        s.dec=nn.Sequential(up(256,128),up(128,64),up(64,32),nn.Conv2d(32,1,1))
    def forward(s,x): return s.dec(s.enc(x))
base=models.CellposeModel(gpu=True, pretrained_model="cpsam_models/cpsam_v2")
net=Det(base.net.encoder.float()).to(dev); net.load_state_dict(torch.load(f"{ROOT}/pointdet.pt")); net.eval()
TIL=[(0,0),(0,256),(256,0),(256,256)]
@torch.no_grad()
def detect(img):
    xs=[np.stack([img[ry:ry+256,rx:rx+256]]*3,0) for (ry,rx) in TIL]
    x=torch.from_numpy(np.stack(xs)).to(dev)
    with torch.autocast("cuda",dtype=torch.bfloat16): hm=net(x).float().cpu().numpy()[:,0]
    allp=[]
    for t,(ry,rx) in enumerate(TIL):
        for (py,px) in peak_local_max(hm[t],min_distance=MIND,threshold_abs=THR,exclude_border=False): allp.append([px+rx,py+ry])
    if not allp: return np.zeros((0,2))
    allp=np.array(allp,float); keep=[]; used=np.zeros(len(allp),bool)
    for i in range(len(allp)):
        if used[i]: continue
        d=np.hypot(allp[:,0]-allp[i,0],allp[:,1]-allp[i,1]); grp=np.where(d<MIND)[0]; used[grp]=True; keep.append(allp[grp].mean(0))
    return np.array(keep)
def cd_of(pts):
    if len(pts)<8: return 0
    ha=ConvexHull(pts).volume*(p2u**2)/1e6; return len(pts)/ha if ha>0 else 0
def mape(p,g): return abs(p-g)/g*100 if g else 0
eA=[];eK=[];ncs=[]
for j,i in enumerate(idx):
    pts=detect(prep(int(i))); g=Y[int(i),0]; ncs.append(len(pts))
    if len(pts)<20: continue
    eA.append(mape(cd_of(pts),g))
    c0=np.median(pts,0); d=np.hypot(pts[:,0]-c0[0],pts[:,1]-c0[1]); K=min(148,len(pts)); kp=pts[np.argsort(d)[:K]]
    eK.append(mape(cd_of(kp),g))
    if j<6: print(f"  i{i}: npts={len(pts)} CDall={cd_of(pts):.0f} CDk={cd_of(kp):.0f} gt={g:.0f}",flush=True)
print(f"\n[point-CD {SPLIT}] n={len(eA)} npts~{np.mean(ncs):.0f}  CD_MAPE(all-hull)={np.mean(eA):.2f}  CD_MAPE(K148)={np.mean(eK):.2f}  (regression CD {6.5})",flush=True)
