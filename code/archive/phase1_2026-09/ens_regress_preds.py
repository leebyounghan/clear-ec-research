"""Ensemble step 0a: per-image regression predictions (CD,CV,HEX in original units) on the
COMMON eval set = first 500 val + first 500 test of the seed-42 split. Saves to ens_common.npz
(ids, split, gt, regress preds). Uses native-512 cache + best baseline checkpoint."""
import os, numpy as np, torch, torch.nn as nn, cv2
from cellpose import models
ROOT="datasets/clear_ec_official"; SZ=512; CROP=256; dev="cuda"
meta=np.load(f"{ROOT}/cache_native_meta_{SZ}.npz",allow_pickle=True)
Y=meta["Y"].astype(np.float32); N=int(meta["N"]); ids=meta["ids"]
imgs=np.array(np.memmap(f"{ROOT}/img_native_{SZ}.dat",dtype=np.uint8,mode='r',shape=(N,SZ,SZ)))
rng=np.random.default_rng(42); perm=rng.permutation(N); te,va,tr=perm[:1000],perm[1000:2000],perm[2000:]
ymean=Y[tr].mean(0)
cval=va[:500]; ctest=te[:500]                    # common set indices
common=np.concatenate([cval,ctest]); split=np.array(["val"]*500+["test"]*500)
def norm99(u):
    lo,hi=np.percentile(u,(1,99)); return np.clip((u.astype(np.float32)-lo)/(hi-lo+1e-6),0,1)
C5=[(0,0),(0,256),(256,0),(256,256),(128,128)]
class Net(nn.Module):
    def __init__(s,enc):
        super().__init__(); s.enc=enc; s.pool=nn.AdaptiveAvgPool2d(1); s.ln=nn.LayerNorm(256)
        def h(): return nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Dropout(.2),nn.Linear(128,1))
        s.hCD,s.hCV,s.hHEX=h(),h(),h()
    def forward(s,x):
        g=s.ln(s.pool(s.enc(x)).flatten(1)); return torch.cat([s.hCD(g),s.hCV(g),s.hHEX(g)],1)+1.0
base=models.CellposeModel(gpu=True, pretrained_model="cpsam_models/cpsam_v2")
net=Net(base.net.encoder.float()).to(dev)
net.load_state_dict(torch.load(f"{ROOT}/regress_ft_best_base.pt")); net.eval()
preds=np.zeros((len(common),3),np.float32); gt=Y[common]
with torch.no_grad():
    for s in range(0,len(common),16):
        b=common[s:s+16]
        x=np.concatenate([np.stack([np.stack([norm99(imgs[i][r:r+CROP,c:c+CROP])]*3,0) for r,c in C5]) for i in b],0)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            p=net(torch.from_numpy(x).to(dev)).float().view(len(b),5,3).mean(1)
        preds[s:s+len(b)]=(p.cpu().numpy()*ymean)
np.savez(f"{ROOT}/ens_common.npz", ids=ids[common], split=split, gt=gt, regress=preds, idx=common, ymean=ymean)
mpe=np.mean(np.abs(preds-gt)/gt*100,0)
print(f"regress on common: CD{mpe[0]:.1f} CV{mpe[1]:.1f} HEX{mpe[2]:.1f} OV{mpe.mean():.1f}  (saved ens_common.npz, {len(common)} imgs)")
