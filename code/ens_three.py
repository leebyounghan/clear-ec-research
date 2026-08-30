"""3-model ensemble with per-model preprocessing (robust_long/dldl no-CLAHE, clahe_long CLAHE).
Average real-unit predictions. Report combos; select on VAL, report TEST (held-out)."""
import os, json, numpy as np, torch, torch.nn as nn, cv2
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
_cl=cv2.createCLAHE(2.0,(8,8))
def norm99(u,clahe):
    if clahe: u=_cl.apply(np.clip(u,0,255).astype(np.uint8)).astype(np.float32)
    lo2,hi2=np.percentile(u,(1,99)); return np.clip((u.astype(np.float32)-lo2)/(hi2-lo2+1e-6),0,1)
C5=[(0,0),(0,256),(256,0),(256,256),(128,128)]
def evc(i,clahe): return np.stack([np.stack([norm99(imgs[i][r:r+CROP,c:c+CROP],clahe)]*3,0) for r,c in C5])
class ScalarNet(nn.Module):
    def __init__(s,enc):
        super().__init__(); s.enc=enc; s.pool=nn.AdaptiveAvgPool2d(1); s.ln=nn.LayerNorm(256)
        def h(): return nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Dropout(.2),nn.Linear(128,1))
        s.hCD,s.hCV,s.hHEX=h(),h(),h()
        def up(i,o): return nn.Sequential(nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),nn.Conv2d(i,o,3,1,1),nn.GroupNorm(8,o),nn.GELU())
        s.dec=nn.Sequential(up(256,128),up(128,64),up(64,32),nn.Conv2d(32,1,1))
    def forward(s,x): f=s.enc(x); g=s.ln(s.pool(f).flatten(1)); return torch.cat([s.hCD(g),s.hCV(g),s.hHEX(g)],1)
class BinNet(nn.Module):
    def __init__(s,enc):
        super().__init__(); s.enc=enc; s.pool=nn.AdaptiveAvgPool2d(1); s.ln=nn.LayerNorm(256)
        def h(): return nn.Sequential(nn.Linear(256,256),nn.GELU(),nn.Dropout(.2),nn.Linear(256,B))
        s.hCD,s.hCV,s.hHEX=h(),h(),h()
    def forward(s,x): f=s.enc(x); g=s.ln(s.pool(f).flatten(1)); return torch.stack([s.hCD(g),s.hCV(g),s.hHEX(g)],2)
base=models.CellposeModel(gpu=True, pretrained_model="cpsam_models/cpsam_v2")
def p_scalar(net,idx,clahe):
    net.eval(); out=np.zeros((len(idx),3),np.float32)
    with torch.no_grad():
        for s in range(0,len(idx),BS):
            b=idx[s:s+BS]; x=np.concatenate([evc(i,clahe) for i in b],0)
            with torch.autocast("cuda",dtype=torch.bfloat16): p=net(torch.from_numpy(x).to(dev)).float().view(len(b),5,3).mean(1)
            out[s:s+len(b)]=torch.exp(p*ls_t+lm_t).cpu().numpy()
    return out
def p_bin(net,idx,clahe):
    net.eval(); out=np.zeros((len(idx),3),np.float32)
    with torch.no_grad():
        for s in range(0,len(idx),BS):
            b=idx[s:s+BS]; x=np.concatenate([evc(i,clahe) for i in b],0)
            with torch.autocast("cuda",dtype=torch.bfloat16): lg=net(torch.from_numpy(x).to(dev)).float().view(len(b),5,B,3).mean(1)
            pr=torch.softmax(lg,1); out[s:s+len(b)]=torch.exp((pr*cent_t[None]).sum(1)).cpu().numpy()
    return out
def mape(P,idx): return np.mean(np.abs(P-Y[idx])/Y[idx]*100,0)
# compute each model's preds (fresh encoder each to avoid shared-weight overwrite)
P={}
n1=ScalarNet(base.net.encoder.float()).to(dev); n1.load_state_dict(torch.load(f"{ROOT}/regress_ft_best_robust_long.pt"))
P["rl_v"]=p_scalar(n1,va,0); P["rl_t"]=p_scalar(n1,te,0)
n2=ScalarNet(base.net.encoder.float()).to(dev); n2.load_state_dict(torch.load(f"{ROOT}/regress_ft_best_clahe_long.pt"))
P["cl_v"]=p_scalar(n2,va,1); P["cl_t"]=p_scalar(n2,te,1)
n3=BinNet(base.net.encoder.float()).to(dev); n3.load_state_dict(torch.load(f"{ROOT}/regress_ft_best_dldl.pt"))
P["dl_v"]=p_bin(n3,va,0); P["dl_t"]=p_bin(n3,te,0)
combos={"clahe":["cl"],"clahe+rl":["cl","rl"],"clahe+dldl":["cl","dl"],"all3":["cl","rl","dl"]}
print(f"{'combo':14} {'VAL_OV':>7} {'TEST_OV':>8}")
res={}
for nm,ks in combos.items():
    Ev=np.mean([P[f"{k}_v"] for k in ks],0); Et=np.mean([P[f"{k}_t"] for k in ks],0)
    mv=mape(Ev,va); mt=mape(Et,te); res[nm]={"val":float(mv.mean()),"test":float(mt.mean())}
    print(f"{nm:14} {mv.mean():7.2f} {mt.mean():8.2f}  (test CD{mt[0]:.1f} CV{mt[1]:.1f} HEX{mt[2]:.1f})",flush=True)
best=min(res.items(),key=lambda kv:kv[1]["val"])
print(f"\nBEST by VAL: {best[0]} -> TEST {best[1]['test']:.2f}",flush=True)
