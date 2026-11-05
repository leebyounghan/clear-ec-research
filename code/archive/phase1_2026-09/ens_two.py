"""Ensemble the log-Huber scalar model (robust_long) + DLDL bin model (dldl): different head
architectures -> handle each separately, convert to REAL units, average per-image. Select on VAL,
report TEST (held-out)."""
import os, json, numpy as np, torch, torch.nn as nn, math
from cellpose import models
ROOT="datasets/clear_ec_official"; SZ=512; CROP=256; dev="cuda"; BS=16; B=50
meta=np.load(f"{ROOT}/cache_native_meta_{SZ}.npz",allow_pickle=True)
Y=meta["Y"].astype(np.float32); N=int(meta["N"])
imgs=np.array(np.memmap(f"{ROOT}/img_native_{SZ}.dat",dtype=np.uint8,mode='r',shape=(N,SZ,SZ)))
rng=np.random.default_rng(42); perm=rng.permutation(N); te,va,tr=perm[:1000],perm[1000:2000],perm[2000:]
logY=np.log(np.clip(Y,1e-6,None)); lm=logY[tr].mean(0); ls=logY[tr].std(0)
ls_t=torch.tensor(ls,device=dev); lm_t=torch.tensor(lm,device=dev)
lo=np.percentile(logY[tr],0.5,axis=0); hi=np.percentile(logY[tr],99.5,axis=0)
centers=np.stack([np.linspace(lo[j],hi[j],B) for j in range(3)],1); cent_t=torch.tensor(centers,device=dev,dtype=torch.float32)
def norm99(u):
    lo2,hi2=np.percentile(u,(1,99)); return np.clip((u.astype(np.float32)-lo2)/(hi2-lo2+1e-6),0,1)
C5=[(0,0),(0,256),(256,0),(256,256),(128,128)]
def evalcrops(i): return np.stack([np.stack([norm99(imgs[i][r:r+CROP,c:c+CROP])]*3,0) for r,c in C5])
class ScalarNet(nn.Module):
    def __init__(s,enc):
        super().__init__(); s.enc=enc; s.pool=nn.AdaptiveAvgPool2d(1); s.ln=nn.LayerNorm(256)
        def h(): return nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Dropout(.2),nn.Linear(128,1))
        s.hCD,s.hCV,s.hHEX=h(),h(),h()
        def up(i,o): return nn.Sequential(nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),nn.Conv2d(i,o,3,1,1),nn.GroupNorm(8,o),nn.GELU())
        s.dec=nn.Sequential(up(256,128),up(128,64),up(64,32),nn.Conv2d(32,1,1))
    def forward(s,x):
        f=s.enc(x); g=s.ln(s.pool(f).flatten(1)); return torch.cat([s.hCD(g),s.hCV(g),s.hHEX(g)],1)
class BinNet(nn.Module):
    def __init__(s,enc):
        super().__init__(); s.enc=enc; s.pool=nn.AdaptiveAvgPool2d(1); s.ln=nn.LayerNorm(256)
        def h(): return nn.Sequential(nn.Linear(256,256),nn.GELU(),nn.Dropout(.2),nn.Linear(256,B))
        s.hCD,s.hCV,s.hHEX=h(),h(),h()
    def forward(s,x):
        f=s.enc(x); g=s.ln(s.pool(f).flatten(1)); return torch.stack([s.hCD(g),s.hCV(g),s.hHEX(g)],2)
base=models.CellposeModel(gpu=True, pretrained_model="cpsam_models/cpsam_v2")
def preds_scalar(net,idx):
    net.eval(); out=np.zeros((len(idx),3),np.float32)
    with torch.no_grad():
        for s in range(0,len(idx),BS):
            b=idx[s:s+BS]; x=np.concatenate([evalcrops(i) for i in b],0)
            with torch.autocast("cuda",dtype=torch.bfloat16): p=net(torch.from_numpy(x).to(dev)).float().view(len(b),5,3).mean(1)
            out[s:s+len(b)]=torch.exp(p*ls_t+lm_t).cpu().numpy()
    return out
def preds_bin(net,idx):
    net.eval(); out=np.zeros((len(idx),3),np.float32)
    with torch.no_grad():
        for s in range(0,len(idx),BS):
            b=idx[s:s+BS]; x=np.concatenate([evalcrops(i) for i in b],0)
            with torch.autocast("cuda",dtype=torch.bfloat16): lg=net(torch.from_numpy(x).to(dev)).float().view(len(b),5,B,3).mean(1)
            pr=torch.softmax(lg,1); e=(pr*cent_t[None]).sum(1); out[s:s+len(b)]=torch.exp(e).cpu().numpy()
    return out
def mape(P,idx): return np.mean(np.abs(P-Y[idx])/Y[idx]*100,0)
# NOTE: sn/bn must NOT share the encoder object (loading bn would overwrite sn's weights).
# Compute scalar preds fully, THEN load dldl into the (same) encoder and compute bin preds.
sn=ScalarNet(base.net.encoder.float()).to(dev); sn.load_state_dict(torch.load(f"{ROOT}/regress_ft_best_robust_long.pt"))
Sv=preds_scalar(sn,va); St=preds_scalar(sn,te)
bn=BinNet(base.net.encoder.float()).to(dev); bn.load_state_dict(torch.load(f"{ROOT}/regress_ft_best_dldl.pt"))
Dv=preds_bin(bn,va); Dt=preds_bin(bn,te)
for nm,P,idx in [("robust_long val",Sv,va),("dldl val",Dv,va)]:
    m=mape(P,idx); print(f"{nm:16} CD{m[0]:.2f} CV{m[1]:.2f} HEX{m[2]:.2f} OV{m.mean():.2f}",flush=True)
Ev=(Sv+Dv)/2; Et=(St+Dt)/2; mv=mape(Ev,va); mt=mape(Et,te)
print(f"\n=== ENSEMBLE (robust_long + dldl) ===")
print(f"VAL  CD{mv[0]:.2f} CV{mv[1]:.2f} HEX{mv[2]:.2f} OV{mv.mean():.2f}")
print(f"TEST CD{mt[0]:.2f} CV{mt[1]:.2f} HEX{mt[2]:.2f} OV{mt.mean():.2f}",flush=True)
json.dump({"val":mv.tolist(),"test":mt.tolist()},open(f"{ROOT}/ens_two_results.json","w"),indent=1)
