"""Stage A (clearec env): run trained point detector on a diverse sample of the 9000 (tr split only,
no val/test leakage). 512-cache image -> CLAHE+minmax -> 2x2 tiles of 256 -> heatmap -> peaks ->
merged 512-coord points. Saves points_9000.npz {idx: Nx2 (x,y)} for the SAM pipeline (Stage B)."""
import os, numpy as np, torch, torch.nn as nn, cv2
from skimage.feature import peak_local_max
from cellpose import models
ROOT="datasets/clear_ec_official"; SZ=512; dev="cuda"
NSAMP=int(os.environ.get("NSAMP","600")); MIND=int(os.environ.get("MIND","7")); THR=float(os.environ.get("THR","0.25"))
meta=np.load(f"{ROOT}/cache_native_meta_{SZ}.npz",allow_pickle=True); N=int(meta["N"])
raw=np.memmap(f"{ROOT}/img_native_{SZ}.dat",dtype=np.uint8,mode='r',shape=(N,SZ,SZ))
rng=np.random.default_rng(42); perm=rng.permutation(N); tr=perm[2000:]
sel=rng.choice(tr, size=min(NSAMP,len(tr)), replace=False)
_cl=cv2.createCLAHE(2.0,(8,8))
def prep(i):
    a=_cl.apply(np.ascontiguousarray(raw[i])).astype(np.float32)
    a=cv2.normalize(a,None,0,255,cv2.NORM_MINMAX)
    lo,hi=np.percentile(a,(1,99)); return np.clip((a-lo)/(hi-lo+1e-6),0,1).astype(np.float32)
class Det(nn.Module):
    def __init__(s,enc):
        super().__init__(); s.enc=enc
        def up(i,o): return nn.Sequential(nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),nn.Conv2d(i,o,3,1,1),nn.GroupNorm(8,o),nn.GELU())
        s.dec=nn.Sequential(up(256,128),up(128,64),up(64,32),nn.Conv2d(32,1,1))
    def forward(s,x): return s.dec(s.enc(x))
base=models.CellposeModel(gpu=True, pretrained_model="cpsam_models/cpsam_v2")
net=Det(base.net.encoder.float()).to(dev); net.load_state_dict(torch.load(f"{ROOT}/pointdet.pt")); net.eval()
print(f"pointdet loaded; detecting on {len(sel)} tr images",flush=True)
TIL=[(0,0),(0,256),(256,0),(256,256)]
@torch.no_grad()
def detect(img):  # img 512x512 float01
    allp=[]
    xs=[];
    for (ry,rx) in TIL: xs.append(np.stack([img[ry:ry+256,rx:rx+256]]*3,0))
    x=torch.from_numpy(np.stack(xs)).to(dev)
    with torch.autocast("cuda",dtype=torch.bfloat16): hm=net(x).float().cpu().numpy()[:,0]  # MSE-regressed [0,1] heatmap
    for t,(ry,rx) in enumerate(TIL):
        pk=peak_local_max(hm[t], min_distance=MIND, threshold_abs=THR, exclude_border=False)
        for (py,px) in pk: allp.append([px+rx, py+ry])
    if not allp: return np.zeros((0,2))
    allp=np.array(allp,float)
    # dedup within MIND across tile seams
    keep=[]; used=np.zeros(len(allp),bool)
    for i in range(len(allp)):
        if used[i]: continue
        d=np.hypot(allp[:,0]-allp[i,0],allp[:,1]-allp[i,1]); grp=np.where(d<MIND)[0]
        used[grp]=True; keep.append(allp[grp].mean(0))
    return np.array(keep)
out={}
for j,i in enumerate(sel):
    img=prep(i); pts=detect(img); out[str(int(i))]=pts
    if j<10 or j%100==0: print(f"  {j}: idx{i} npts={len(pts)}",flush=True)
np.savez(f"{ROOT}/points_9000.npz", **{k:v for k,v in out.items()})
print(f"DONE saved points for {len(out)} images -> points_9000.npz",flush=True)
