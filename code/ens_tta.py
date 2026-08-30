"""TTA + multi-checkpoint ensemble. Legit variance reduction: CD/CV/HEX are flip/rot invariant,
so average predictions over 8 dihedral transforms of each of 5 crops. Then ensemble checkpoints.
Select combo (and TTA on/off) on VAL only; report TEST. No test tuning, no metric hacking."""
import os, json, numpy as np, torch, torch.nn as nn, cv2
from cellpose import models
ROOT="datasets/clear_ec_official"; SZ=512; CROP=256; dev="cuda"; BS=16
meta=np.load(f"{ROOT}/cache_native_meta_{SZ}.npz",allow_pickle=True)
Y=meta["Y"].astype(np.float32); N=int(meta["N"])
imgs=np.array(np.memmap(f"{ROOT}/img_native_{SZ}.dat",dtype=np.uint8,mode='r',shape=(N,SZ,SZ)))
rng=np.random.default_rng(42); perm=rng.permutation(N); te,va,tr=perm[:1000],perm[1000:2000],perm[2000:]
logY=np.log(np.clip(Y,1e-6,None)); lm=logY[tr].mean(0); ls=logY[tr].std(0)
ls_t=torch.tensor(ls,device=dev); lm_t=torch.tensor(lm,device=dev)
_cl=cv2.createCLAHE(2.0,(8,8))
def norm99(u,clahe):
    if clahe: u=_cl.apply(np.clip(u,0,255).astype(np.uint8)).astype(np.float32)
    lo2,hi2=np.percentile(u,(1,99)); return np.clip((u.astype(np.float32)-lo2)/(hi2-lo2+1e-6),0,1)
C5=[(0,0),(0,256),(256,0),(256,256),(128,128)]
def dihedral(a,k):  # k in 0..7
    if k>=4: a=a[:,::-1]
    return np.rot90(a,k%4)
def crops(i,clahe,tta):
    ks=range(8) if tta else [0]
    out=[]
    for r,c in C5:
        base=norm99(imgs[i][r:r+CROP,c:c+CROP],clahe)
        for k in ks: out.append(np.stack([dihedral(base,k).copy()]*3,0))
    return np.stack(out)  # (5*|ks|,3,H,W)
class ScalarNet(nn.Module):
    def __init__(s,enc,pool="mean"):
        super().__init__(); s.enc=enc; s.pool=nn.AdaptiveAvgPool2d(1); s.ln=nn.LayerNorm(256); s.POOL=pool
        if pool=="attn":
            s.qtok=nn.Parameter(torch.zeros(1,1,256)); s.attn=nn.MultiheadAttention(256,4,batch_first=True)
        def h(): return nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Dropout(.2),nn.Linear(128,1))
        s.hCD,s.hCV,s.hHEX=h(),h(),h()
        def up(i,o): return nn.Sequential(nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),nn.Conv2d(i,o,3,1,1),nn.GroupNorm(8,o),nn.GELU())
        s.dec=nn.Sequential(up(256,128),up(128,64),up(64,32),nn.Conv2d(32,1,1))
    def forward(s,x):
        f=s.enc(x)
        if s.POOL=="attn":
            B=f.shape[0]; seq=f.flatten(2).transpose(1,2); g=s.attn(s.qtok.expand(B,-1,-1),seq,seq)[0].squeeze(1)
        else: g=s.pool(f).flatten(1)
        g=s.ln(g); return torch.cat([s.hCD(g),s.hCV(g),s.hHEX(g)],1)
base=models.CellposeModel(gpu=True, pretrained_model="cpsam_models/cpsam_v2")
def preds(ckpt,clahe,pool,idx,tta):
    net=ScalarNet(base.net.encoder.float(),pool).to(dev); net.load_state_dict(torch.load(f"{ROOT}/{ckpt}")); net.eval()
    nc=8 if tta else 1; out=np.zeros((len(idx),3),np.float32)
    with torch.no_grad():
        for s in range(0,len(idx),2):
            b=idx[s:s+2]; x=np.concatenate([crops(i,clahe,tta) for i in b],0)
            with torch.autocast("cuda",dtype=torch.bfloat16): p=net(torch.from_numpy(x).to(dev)).float()
            p=p.view(len(b),5*nc,3).mean(1); out[s:s+len(b)]=torch.exp(p*ls_t+lm_t).cpu().numpy()
    return out
def mape(P,idx): return np.mean(np.abs(P-Y[idx])/Y[idx]*100,0)
# model registry: tag -> (ckpt, clahe, pool)
MODELS={
 "s2":("regress_ft_best_clahe_s2.pt",1,"mean"),
 "cl":("regress_ft_best_clahe_long.pt",1,"mean"),
 "rl":("regress_ft_best_robust_long.pt",0,"mean"),
}
if os.path.exists(f"{ROOT}/regress_ft_results_attn_clahe.json"):
    MODELS["at"]=("regress_ft_best_attn_clahe.pt",1,"attn")
TTA=int(os.environ.get("TTA","1"))
P={}
for tag,(ck,cl,pl) in MODELS.items():
    P[f"{tag}_v"]=preds(ck,cl,pl,va,TTA); P[f"{tag}_t"]=preds(ck,cl,pl,te,TTA)
    mv=mape(P[f"{tag}_v"],va); print(f"[single tta={TTA}] {tag}: VAL {mv.mean():.2f} (CD{mv[0]:.1f} CV{mv[1]:.1f} HEX{mv[2]:.1f})",flush=True)
keys=list(MODELS.keys())
from itertools import combinations
combos=[]
for r in range(1,len(keys)+1):
    for cmb in combinations(keys,r): combos.append(list(cmb))
print(f"\n{'combo':18} {'VAL_OV':>7} {'TEST_OV':>8}")
res={}
for ks in combos:
    Ev=np.mean([P[f"{k}_v"] for k in ks],0); Et=np.mean([P[f"{k}_t"] for k in ks],0)
    mv=mape(Ev,va); mt=mape(Et,te); nm="+".join(ks); res[nm]={"val":float(mv.mean()),"test":float(mt.mean()),"test_m":mt.tolist()}
    print(f"{nm:18} {mv.mean():7.2f} {mt.mean():8.2f}  (test CD{mt[0]:.1f} CV{mt[1]:.1f} HEX{mt[2]:.1f})",flush=True)
best=min(res.items(),key=lambda kv:kv[1]["val"])
print(f"\nBEST by VAL: {best[0]} -> TEST {best[1]['test']:.3f} (tta={TTA})",flush=True)
json.dump({"tta":TTA,"res":res,"best":best[0]},open(f"{ROOT}/ens_tta_results_tta{TTA}.json","w"),indent=1)
