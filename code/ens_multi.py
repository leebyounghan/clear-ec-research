"""Ensemble multiple log-Huber regression checkpoints. Loads each regress_ft_best_{TAG}.pt,
computes per-image 5-crop predictions in REAL units (exp of log-pred), averages across models.
Model selection on VAL; TEST reported only (held-out, no tuning on it). TAGS env = comma list."""
import os, json, numpy as np, torch, torch.nn as nn
from cellpose import models
ROOT="datasets/clear_ec_official"; SZ=512; CROP=256; dev="cuda"; BS=16
TAGS=os.environ.get("TAGS","robust_long").split(",")
meta=np.load(f"{ROOT}/cache_native_meta_{SZ}.npz",allow_pickle=True)
Y=meta["Y"].astype(np.float32); N=int(meta["N"])
imgs=np.array(np.memmap(f"{ROOT}/img_native_{SZ}.dat",dtype=np.uint8,mode='r',shape=(N,SZ,SZ)))
rng=np.random.default_rng(42); perm=rng.permutation(N); te,va,tr=perm[:1000],perm[1000:2000],perm[2000:]
logY=np.log(np.clip(Y,1e-6,None)); lm=logY[tr].mean(0); ls=logY[tr].std(0)
ls_t=torch.tensor(ls,device=dev); lm_t=torch.tensor(lm,device=dev)
def norm99(u):
    lo,hi=np.percentile(u,(1,99)); return np.clip((u.astype(np.float32)-lo)/(hi-lo+1e-6),0,1)
C5=[(0,0),(0,256),(256,0),(256,256),(128,128)]
class Net(nn.Module):
    def __init__(s, enc):
        super().__init__(); s.enc=enc; s.pool=nn.AdaptiveAvgPool2d(1); s.ln=nn.LayerNorm(256)
        def h(): return nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Dropout(.2),nn.Linear(128,1))
        s.hCD,s.hCV,s.hHEX=h(),h(),h()
        def up(i,o): return nn.Sequential(nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),nn.Conv2d(i,o,3,1,1),nn.GroupNorm(8,o),nn.GELU())
        s.dec=nn.Sequential(up(256,128),up(128,64),up(64,32),nn.Conv2d(32,1,1))
    def forward(s,x):
        f=s.enc(x); g=s.ln(s.pool(f).flatten(1)); return torch.cat([s.hCD(g),s.hCV(g),s.hHEX(g)],1)
base=models.CellposeModel(gpu=True, pretrained_model="cpsam_models/cpsam_v2")
net=Net(base.net.encoder.float()).to(dev)
def preds_real(idx):
    net.eval(); out=np.zeros((len(idx),3),np.float32)
    with torch.no_grad():
        for s in range(0,len(idx),BS):
            b=idx[s:s+BS]
            x=np.concatenate([np.stack([np.stack([norm99(imgs[i][r:r+CROP,c:c+CROP])]*3,0) for r,c in C5]) for i in b],0)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                p=net(torch.from_numpy(x).to(dev)).float().view(len(b),5,3).mean(1)
            out[s:s+len(b)]=torch.exp(p*ls_t+lm_t).cpu().numpy()
    return out
def mape(P,idx): return np.mean(np.abs(P-Y[idx])/Y[idx]*100,0)
Pv=np.zeros((len(va),3)); Pt=np.zeros((len(te),3)); got=[]
for tag in TAGS:
    ck=f"{ROOT}/regress_ft_best_{tag.strip()}.pt"
    if not os.path.exists(ck): print(f"skip {tag} (no ckpt)",flush=True); continue
    net.load_state_dict(torch.load(ck)); vv=preds_real(va); tt=preds_real(te); Pv+=vv; Pt+=tt; got.append(tag.strip())
    m=mape(vv,va); print(f"{tag.strip():14} val CD{m[0]:.2f} CV{m[1]:.2f} HEX{m[2]:.2f} OV{m.mean():.2f}",flush=True)
n=len(got); Pv/=n; Pt/=n
mv=mape(Pv,va); mt=mape(Pt,te)
print(f"\n=== ENSEMBLE ({n} models: {got}) ===")
print(f"VAL  CD{mv[0]:.2f} CV{mv[1]:.2f} HEX{mv[2]:.2f} OV{mv.mean():.2f}")
print(f"TEST CD{mt[0]:.2f} CV{mt[1]:.2f} HEX{mt[2]:.2f} OV{mt.mean():.2f}")
json.dump({"models":got,"val":mv.tolist(),"test":mt.tolist()},open(f"{ROOT}/ens_multi_results.json","w"),indent=1)
