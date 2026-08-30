"""Final max-diversity ensemble. Members differ in preprocessing (per-crop CLAHE / no-CLAHE /
full-image CLAHE cache) AND architecture (mean / attn / meanstd pool) -> maximally decorrelated.
8-dihedral TTA per crop. Search all combos, select by VAL, report TEST. No test tuning."""
import os, json, numpy as np, torch, torch.nn as nn, cv2
from itertools import combinations
from cellpose import models
ROOT="datasets/clear_ec_official"; SZ=512; CROP=256; dev="cuda"
meta=np.load(f"{ROOT}/cache_native_meta_{SZ}.npz",allow_pickle=True); Y=meta["Y"].astype(np.float32); N=int(meta["N"])
native=np.array(np.memmap(f"{ROOT}/img_native_{SZ}.dat",dtype=np.uint8,mode='r',shape=(N,SZ,SZ)))
clahe_full=np.array(np.memmap(f"{ROOT}/img_clahe_{SZ}.dat",dtype=np.uint8,mode='r',shape=(N,SZ,SZ)))
rng=np.random.default_rng(42); perm=rng.permutation(N); te,va,tr=perm[:1000],perm[1000:2000],perm[2000:]
logY=np.log(np.clip(Y,1e-6,None)); lm=logY[tr].mean(0); ls=logY[tr].std(0)
ls_t=torch.tensor(ls,device=dev); lm_t=torch.tensor(lm,device=dev)
_cl=cv2.createCLAHE(2.0,(8,8))
def norm99_raw(u,clahe):
    if clahe: u=_cl.apply(np.clip(u,0,255).astype(np.uint8)).astype(np.float32)
    lo,hi=np.percentile(u,(1,99)); return np.clip((u.astype(np.float32)-lo)/(hi-lo+1e-6),0,1)
def norm99_full(u):  # already CLAHE'd cache
    lo,hi=np.percentile(u,(1,99)); return np.clip((u.astype(np.float32)-lo)/(hi-lo+1e-6),0,1)
C5=[(0,0),(0,256),(256,0),(256,256),(128,128)]
def dih(a,k): return np.rot90(a[:,::-1] if k>=4 else a, k%4)
class ScalarNet(nn.Module):
    def __init__(s,enc,pool):
        super().__init__(); s.enc=enc; s.pool=nn.AdaptiveAvgPool2d(1); D=512 if pool=="meanstd" else 256; s.ln=nn.LayerNorm(D); s.P=pool
        if pool=="attn": s.qtok=nn.Parameter(torch.zeros(1,1,256)); s.attn=nn.MultiheadAttention(256,4,batch_first=True)
        def h(): return nn.Sequential(nn.Linear(D,128),nn.GELU(),nn.Dropout(.2),nn.Linear(128,1))
        s.hCD,s.hCV,s.hHEX=h(),h(),h()
        def up(i,o): return nn.Sequential(nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),nn.Conv2d(i,o,3,1,1),nn.GroupNorm(8,o),nn.GELU())
        s.dec=nn.Sequential(up(256,128),up(128,64),up(64,32),nn.Conv2d(32,1,1))
    def forward(s,x):
        f=s.enc(x); B=f.shape[0]
        if s.P=="attn": seq=f.flatten(2).transpose(1,2); g=s.attn(s.qtok.expand(B,-1,-1),seq,seq)[0].squeeze(1)
        elif s.P=="meanstd": v=f.flatten(2); g=torch.cat([v.mean(-1),v.std(-1)],1)
        else: g=s.pool(f).flatten(1)
        g=s.ln(g); return torch.cat([s.hCD(g),s.hCV(g),s.hHEX(g)],1)
base=models.CellposeModel(gpu=True, pretrained_model="cpsam_models/cpsam_v2")
def crops(i,src,clahe,tta):
    ks=range(8) if tta else [0]; out=[]
    img = clahe_full[i] if src=="full" else native[i]
    for r,c in C5:
        b = norm99_full(img[r:r+CROP,c:c+CROP]) if src=="full" else norm99_raw(img[r:r+CROP,c:c+CROP],clahe)
        for k in ks: out.append(np.stack([dih(b,k).copy()]*3,0))
    return np.stack(out)
def preds(ckpt,src,clahe,pool,idx,tta=True):
    net=ScalarNet(base.net.encoder.float(),pool).to(dev); net.load_state_dict(torch.load(f"{ROOT}/{ckpt}"),strict=False); net.eval()
    nc=8 if tta else 1; out=np.zeros((len(idx),3),np.float32)
    with torch.no_grad():
        for s in range(0,len(idx),2):
            b=idx[s:s+2]; x=np.concatenate([crops(i,src,clahe,tta) for i in b],0)
            with torch.autocast("cuda",dtype=torch.bfloat16): p=net(torch.from_numpy(x).to(dev)).float()
            p=p.view(len(b),5*nc,3).mean(1); out[s:s+len(b)]=torch.exp(p*ls_t+lm_t).cpu().numpy()
    del net; torch.cuda.empty_cache(); return out
def mape(P,idx): return np.mean(np.abs(P-Y[idx])/Y[idx]*100,0)
# tag -> (ckpt, src, clahe, pool)
MODELS={
 "s2":("regress_ft_best_clahe_s2.pt","raw",1,"mean"),
 "cl":("regress_ft_best_clahe_long.pt","raw",1,"mean"),
 "rl":("regress_ft_best_robust_long.pt","raw",0,"mean"),
 "at":("regress_ft_best_attn_clahe.pt","raw",1,"attn"),
 "mt":("regress_ft_best_mtl_l05.pt","full",0,"mean"),
 "sp":("regress_ft_best_stdpool_ms.pt","full",0,"meanstd"),
}
P={}
for tag,(ck,src,cl,pool) in MODELS.items():
    P[f"{tag}_v"]=preds(ck,src,cl,pool,va); P[f"{tag}_t"]=preds(ck,src,cl,pool,te)
    mv=mape(P[f"{tag}_v"],va); print(f"[{tag}] VAL {mv.mean():.2f} (CV{mv[1]:.1f} HEX{mv[2]:.1f})",flush=True)
keys=list(MODELS.keys()); best=None
for r in range(1,len(keys)+1):
    for cmb in combinations(keys,r):
        Ev=np.mean([P[f"{k}_v"] for k in cmb],0); Et=np.mean([P[f"{k}_t"] for k in cmb],0)
        mv=mape(Ev,va).mean(); mt=mape(Et,te)
        if best is None or mv<best[1]: best=("+".join(cmb),mv,float(mt.mean()),mt.tolist())
print(f"\nBEST by VAL: {best[0]} -> VAL {best[1]:.3f} TEST {best[2]:.3f} (CD{best[3][0]:.1f} CV{best[3][1]:.1f} HEX{best[3][2]:.1f})",flush=True)
json.dump({"best":best[0],"val":best[1],"test":best[2],"test_m":best[3]},open(f"{ROOT}/ens_final_results.json","w"),indent=1)
