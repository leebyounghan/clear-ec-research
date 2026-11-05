"""Ensemble two CLAHE scalar-log-Huber seeds (clahe_long + clahe_s2): both same recipe, different
training stochasticity -> diversity. Separate encoder per model (avoid shared-weight overwrite).
Average real-unit preds. Select on VAL, report TEST (held-out)."""
import os, numpy as np, torch, torch.nn as nn, cv2
from cellpose import models
ROOT="datasets/clear_ec_official"; SZ=512; CROP=256; dev="cuda"; BS=16
meta=np.load(f"{ROOT}/cache_native_meta_{SZ}.npz",allow_pickle=True)
Y=meta["Y"].astype(np.float32); N=int(meta["N"])
imgs=np.array(np.memmap(f"{ROOT}/img_native_{SZ}.dat",dtype=np.uint8,mode='r',shape=(N,SZ,SZ)))
rng=np.random.default_rng(42); perm=rng.permutation(N); te,va,tr=perm[:1000],perm[1000:2000],perm[2000:]
logY=np.log(np.clip(Y,1e-6,None)); lm=logY[tr].mean(0); ls=logY[tr].std(0)
ls_t=torch.tensor(ls,device=dev); lm_t=torch.tensor(lm,device=dev)
_cl=cv2.createCLAHE(2.0,(8,8))
def norm99(u):
    u=_cl.apply(np.clip(u,0,255).astype(np.uint8)).astype(np.float32)
    lo2,hi2=np.percentile(u,(1,99)); return np.clip((u-lo2)/(hi2-lo2+1e-6),0,1)
C5=[(0,0),(0,256),(256,0),(256,256),(128,128)]
def evc(i): return np.stack([np.stack([norm99(imgs[i][r:r+CROP,c:c+CROP])]*3,0) for r,c in C5])
class ScalarNet(nn.Module):
    def __init__(s,enc):
        super().__init__(); s.enc=enc; s.pool=nn.AdaptiveAvgPool2d(1); s.ln=nn.LayerNorm(256)
        def h(): return nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Dropout(.2),nn.Linear(128,1))
        s.hCD,s.hCV,s.hHEX=h(),h(),h()
        def up(i,o): return nn.Sequential(nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),nn.Conv2d(i,o,3,1,1),nn.GroupNorm(8,o),nn.GELU())
        s.dec=nn.Sequential(up(256,128),up(128,64),up(64,32),nn.Conv2d(32,1,1))
    def forward(s,x): f=s.enc(x); g=s.ln(s.pool(f).flatten(1)); return torch.cat([s.hCD(g),s.hCV(g),s.hHEX(g)],1)
base=models.CellposeModel(gpu=True, pretrained_model="cpsam_models/cpsam_v2")
def preds(net,idx):
    net.eval(); out=np.zeros((len(idx),3),np.float32)
    with torch.no_grad():
        for s in range(0,len(idx),BS):
            b=idx[s:s+BS]; x=np.concatenate([evc(i) for i in b],0)
            with torch.autocast("cuda",dtype=torch.bfloat16): p=net(torch.from_numpy(x).to(dev)).float().view(len(b),5,3).mean(1)
            out[s:s+len(b)]=torch.exp(p*ls_t+lm_t).cpu().numpy()
    return out
def mape(P,idx): return np.mean(np.abs(P-Y[idx])/Y[idx]*100,0)
n1=ScalarNet(base.net.encoder.float()).to(dev); n1.load_state_dict(torch.load(f"{ROOT}/regress_ft_best_clahe_long.pt"))
A_v=preds(n1,va); A_t=preds(n1,te)
n2=ScalarNet(base.net.encoder.float()).to(dev); n2.load_state_dict(torch.load(f"{ROOT}/regress_ft_best_clahe_s2.pt"))
B_v=preds(n2,va); B_t=preds(n2,te)
for nm,Pv,Pt in [("clahe_long",A_v,A_t),("clahe_s2",B_v,B_t)]:
    mv=mape(Pv,va); mt=mape(Pt,te); print(f"{nm:12} VAL {mv.mean():.2f}  TEST {mt.mean():.2f}",flush=True)
Ev=(A_v+B_v)/2; Et=(A_t+B_t)/2; mv=mape(Ev,va); mt=mape(Et,te)
print(f"\n=== 2-seed CLAHE ENSEMBLE ===",flush=True)
print(f"VAL  CD{mv[0]:.2f} CV{mv[1]:.2f} HEX{mv[2]:.2f} OV{mv.mean():.2f}",flush=True)
print(f"TEST CD{mt[0]:.2f} CV{mt[1]:.2f} HEX{mt[2]:.2f} OV{mt.mean():.2f}",flush=True)
