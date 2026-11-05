"""Regression predictions on the pool rows (for the regression+fNLA stacking fusion).
Runs clahe_s2 (log-space, C5x8-TTA) on img_native_512 for every pool row -> {row: [CD,CV,HEX]}.
Small batches to fit alongside other GPU jobs. Saves reg_pool_preds.json."""
import os, json, numpy as np, torch, torch.nn as nn, cv2
from cellpose import models
ROOT="datasets/clear_ec_official"; SZ=512; CROP=256; dev="cuda"
LM=np.array([7.919919967651367,-0.9824771285057068,-0.6211431622505188],np.float32)
LS=np.array([0.18971863389015198,0.18813571333885193,0.2748945653438568],np.float32)
_cl=cv2.createCLAHE(2.0,(8,8)); C5=[(0,0),(0,256),(256,0),(256,256),(128,128)]
def dih(a,k): return np.rot90(a[:,::-1] if k>=4 else a,k%4)
def n99(u): u=_cl.apply(np.clip(u,0,255).astype(np.uint8)).astype(np.float32); lo,hi=np.percentile(u,(1,99)); return np.clip((u-lo)/(hi-lo+1e-6),0,1)
cm=np.load(f"{ROOT}/cache_native_meta_512.npz",allow_pickle=True); cids=list(cm["ids"]); Nc=int(cm["N"])
imgs=np.array(np.memmap(f"{ROOT}/img_native_{SZ}.dat",dtype=np.uint8,mode='r',shape=(Nc,SZ,SZ)))
pm=np.load(f"{ROOT}/pool_meta.npz",allow_pickle=True); pids=pm["ids"]
f2row={f:i for i,f in enumerate(cids)}; rows=sorted({f2row[str(p)] for p in pids if str(p) in f2row})
class SN(nn.Module):
    def __init__(s,enc):
        super().__init__(); s.enc=enc; s.pool=nn.AdaptiveAvgPool2d(1); s.ln=nn.LayerNorm(256)
        def h(): return nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Dropout(.2),nn.Linear(128,1))
        s.hCD,s.hCV,s.hHEX=h(),h(),h()
    def forward(s,x): g=s.ln(s.pool(s.enc(x)).flatten(1)); return torch.cat([s.hCD(g),s.hCV(g),s.hHEX(g)],1)
base=models.CellposeModel(gpu=True, pretrained_model="cpsam_models/cpsam_v2")
net=SN(base.net.encoder.float()).to(dev); net.load_state_dict(torch.load(f"{ROOT}/regress_ft_best_clahe_s2.pt"),strict=False); net.eval()
out={}
with torch.no_grad():
    for n,i in enumerate(rows):
        xs=[]
        for r,c in C5:
            w=n99(imgs[i][r:r+CROP,c:c+CROP])
            for k in range(8): xs.append(np.stack([np.ascontiguousarray(dih(w,k))]*3,0))
        x=torch.from_numpy(np.stack(xs)).to(dev)
        with torch.autocast("cuda",dtype=torch.bfloat16): p=net(x).float().mean(0).cpu().numpy()
        out[str(i)]=np.exp(p*LS+LM).tolist()
        if n%300==0: print(f"{n}/{len(rows)}",flush=True)
json.dump(out,open(f"{ROOT}/reg_pool_preds.json","w")); print(f"DONE {len(out)} -> reg_pool_preds.json",flush=True)
