"""Test the user's ROI-matching hypothesis (zero-shot, no retrain): instead of averaging the C5
fixed crops of the central 512, pick the grader-like ROI = the sharpest 256 window (highest Laplacian
focus, mimicking a grader choosing a clear compact region), run regression there (+8-dihedral TTA).
Compare CV/HEX vs the C5 baseline on val. If ROI-selection helps zero-shot, a retrain would help more.
Deployable: focus-based window pick uses no GT/segmentation."""
import os, numpy as np, torch, torch.nn as nn, cv2
from cellpose import models
ROOT="datasets/clear_ec_official"; SZ=512; CROP=256; BS=16; dev="cuda"
TAG=os.environ.get("TAG","clahe_long"); CLAHE=int(os.environ.get("CLAHE","1")); _cl=cv2.createCLAHE(2.0,(8,8))
meta=np.load(f"{ROOT}/cache_native_meta_{SZ}.npz",allow_pickle=True); Y=meta["Y"].astype(np.float32); N=int(meta["N"])
imgs=np.array(np.memmap(f"{ROOT}/img_native_{SZ}.dat",dtype=np.uint8,mode='r',shape=(N,SZ,SZ)))
rng=np.random.default_rng(42); perm=rng.permutation(N); te,va,tr=perm[:1000],perm[1000:2000],perm[2000:]
ymean=Y[tr].mean(0); Yn=Y/ymean; NVAL=int(os.environ.get("NVAL","1000"))
def norm99(u):
    if CLAHE: u=_cl.apply(u.astype(np.uint8))
    lo,hi=np.percentile(u,(1,99)); return np.clip((u.astype(np.float32)-lo)/(hi-lo+1e-6),0,1)
C5=[(0,0),(0,256),(256,0),(256,256),(128,128)]
def dih(a,k): return np.ascontiguousarray(np.rot90(a[:,::-1] if k>=4 else a, k%4))
def best_focus_win(u):  # sharpest 256 window in the 512 (stride 64)
    best=-1; by=bx=128
    for y in range(0,SZ-CROP+1,64):
        for x in range(0,SZ-CROP+1,64):
            f=cv2.Laplacian(u[y:y+CROP,x:x+CROP].astype(np.float32),cv2.CV_32F).var()
            if f>best: best=f; by,bx=y,x
    return by,bx
class Net(nn.Module):
    def __init__(s,enc):
        super().__init__(); s.enc=enc; s.pool=nn.AdaptiveAvgPool2d(1); s.ln=nn.LayerNorm(256)
        def h(): return nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Dropout(.2),nn.Linear(128,1))
        s.hCD,s.hCV,s.hHEX=h(),h(),h()
    def forward(s,x):
        g=s.ln(s.pool(s.enc(x)).flatten(1)); return torch.cat([s.hCD(g),s.hCV(g),s.hHEX(g)],1)+1.0
base=models.CellposeModel(gpu=True, pretrained_model="cpsam_models/cpsam_v2")
net=Net(base.net.encoder.float()).to(dev); net.load_state_dict(torch.load(f"{ROOT}/regress_ft_best_{TAG}.pt"),strict=False); net.eval()
Ytr=torch.tensor(Yn,device=dev)
@torch.no_grad()
def run(crops):  # list of HxW float01 -> mean prediction
    x=torch.from_numpy(np.stack([np.stack([c]*3,0) for c in crops])).to(dev)
    with torch.autocast("cuda",dtype=torch.bfloat16): p=net(x).float().mean(0)
    return p
def eval_mode(idx, mode):
    es=[]
    for i in idx:
        u=imgs[int(i)]
        if mode=="c5": crops=[norm99(u[r:r+CROP,c:c+CROP]) for r,c in C5]
        else:
            by,bx=best_focus_win(u); w=norm99(u[by:by+CROP,bx:bx+CROP]); crops=[dih(w,k) for k in range(8)]  # focus-ROI +8TTA
        p=run(crops).cpu().numpy(); y=Yn[int(i)]
        es.append(np.abs(p-y)/np.clip(y,1e-6,None)*100)
    return np.array(es).mean(0)
idx=va[:NVAL]
c5=eval_mode(idx,"c5"); roi=eval_mode(idx,"roi")
print(f"[{TAG}] VAL n={len(idx)}")
print(f"  C5 baseline : CD{c5[0]:.2f} CV{c5[1]:.2f} HEX{c5[2]:.2f} OV{c5.mean():.2f}")
print(f"  focus-ROI+TTA: CD{roi[0]:.2f} CV{roi[1]:.2f} HEX{roi[2]:.2f} OV{roi.mean():.2f}",flush=True)
