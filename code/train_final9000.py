"""FINAL retrain on ALL images (no holdout) for the submission ensemble.
Recipes via env:
  POOL=mean|attn   ARCH=scalar|spec   EPOCHS  TAG   YSRC=raw|smooth9000
  scalar: ScalarNet, log-space targets (container decodes exp(p*LS+LM))
  spec:   SpecNet (+1.0), ymean-space (container multiplies YMEAN), loss on METRIC only
YSRC=smooth9000 -> Y_slidesmooth9000.npy (slide-mean over ALL frames; legit: only train labels used,
leaderboard test is disjoint). Model selection is impossible without val -> fixed-epoch schedule,
checkpoint saved every 5 epochs + final (container uses final unless stated otherwise)."""
import os, numpy as np, torch, torch.nn as nn, json, time, cv2
from cellpose import models
ROOT="datasets/clear_ec_official"; SZ=512; CROP=256
EPOCHS=int(os.environ.get("EPOCHS","25")); BS=int(os.environ.get("BS","16"))
POOL=os.environ.get("POOL","mean"); ARCH=os.environ.get("ARCH","scalar")
YSRC=os.environ.get("YSRC","raw"); TAG=os.environ.get("TAG","final"); METRIC=int(os.environ.get("METRIC","-1"))
LM=np.array([7.919919967651367,-0.9824771285057068,-0.6211431622505188],np.float32)
LS=np.array([0.18971863389015198,0.18813571333885193,0.2748945653438568],np.float32)
YMEAN=np.array([2795.84130859375,0.3811357915401459,0.545988917350769],np.float32)
_clahe=cv2.createCLAHE(2.0,(8,8)); dev="cuda"
meta=np.load(f"{ROOT}/cache_native_meta_{SZ}.npz",allow_pickle=True)
Y=meta["Y"].astype(np.float32); N=int(meta["N"])
Ysrc=np.load(f"{ROOT}/Y_slidesmooth9000.npy").astype(np.float32) if YSRC=="smooth9000" else Y
imgs=np.array(np.memmap(f"{ROOT}/img_native_{SZ}.dat",dtype=np.uint8,mode='r',shape=(N,SZ,SZ)))
bad=set(np.where((Ysrc<=0).any(1))[0].tolist())
tr=np.array([i for i in range(N) if i not in bad])
print(f"[{TAG}] ARCH={ARCH} POOL={POOL} YSRC={YSRC} EPOCHS={EPOCHS} train {len(tr)} (ALL data, dropped {len(bad)})",flush=True)
if ARCH=="scalar": Z=(np.log(np.clip(Ysrc,1e-6,None))-LM)/LS
else: Z=Ysrc/YMEAN

def norm99(u):
    u=_clahe.apply(np.ascontiguousarray(u))
    lo,hi=np.percentile(u,(1,99)); return np.clip((u.astype(np.float32)-lo)/(hi-lo+1e-6),0,1)
def make(i):
    r,c=np.random.randint(0,SZ-CROP+1),np.random.randint(0,SZ-CROP+1)
    im=norm99(imgs[i][r:r+CROP,c:c+CROP])
    if np.random.rand()<.5: im=im[:,::-1]
    if np.random.rand()<.5: im=im[::-1,:]
    im=np.ascontiguousarray(np.rot90(im,np.random.randint(4)))
    return np.stack([im]*3,0)[None]
def tbatch(idx):
    return torch.from_numpy(np.concatenate([make(i) for i in idx],0)).to(dev)

class ScalarNet(nn.Module):
    def __init__(s,enc,pool):
        super().__init__(); s.enc=enc; s.pool=nn.AdaptiveAvgPool2d(1); s.ln=nn.LayerNorm(256); s.P=pool
        if pool=="attn": s.qtok=nn.Parameter(torch.zeros(1,1,256)); s.attn=nn.MultiheadAttention(256,4,batch_first=True)
        def h():
            m=nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Dropout(.2),nn.Linear(128,1))
            nn.init.zeros_(m[-1].weight); nn.init.zeros_(m[-1].bias); return m
        s.hCD,s.hCV,s.hHEX=h(),h(),h()
    def forward(s,x):
        f=s.enc(x); B=f.shape[0]
        if s.P=="attn": seq=f.flatten(2).transpose(1,2); g=s.attn(s.qtok.expand(B,-1,-1),seq,seq)[0].squeeze(1)
        else: g=s.pool(f).flatten(1)
        g=s.ln(g); return torch.cat([s.hCD(g),s.hCV(g),s.hHEX(g)],1)
class SpecNet(nn.Module):
    def __init__(s,enc):
        super().__init__(); s.enc=enc; s.pool=nn.AdaptiveAvgPool2d(1); s.ln=nn.LayerNorm(256)
        def h():
            m=nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Dropout(.2),nn.Linear(128,1))
            nn.init.zeros_(m[-1].weight); nn.init.zeros_(m[-1].bias); return m
        s.hCD,s.hCV,s.hHEX=h(),h(),h()
    def forward(s,x): g=s.ln(s.pool(s.enc(x)).flatten(1)); return torch.cat([s.hCD(g),s.hCV(g),s.hHEX(g)],1)+1.0
base=models.CellposeModel(gpu=True, pretrained_model="cpsam_models/cpsam_v2")
net=(ScalarNet(base.net.encoder.float(),POOL) if ARCH=="scalar" else SpecNet(base.net.encoder.float())).to(dev)
import torch.utils.checkpoint as cp
for blk in net.enc.blocks:
    blk._of=blk.forward
    blk.forward=(lambda m: (lambda x: cp.checkpoint(m._of, x, use_reentrant=False)))(blk)
heads=list(net.hCD.parameters())+list(net.hCV.parameters())+list(net.hHEX.parameters())
if POOL=="attn" and ARCH=="scalar": heads+= [net.qtok]+list(net.attn.parameters())
opt=torch.optim.AdamW([{"params":net.enc.parameters(),"lr":1e-5},{"params":heads,"lr":1e-3}],weight_decay=1e-4)
Zt=torch.tensor(Z,device=dev)
sched=torch.optim.lr_scheduler.LambdaLR(opt, lambda step: min(1.0,(step+1)/200))
nb=len(tr)//BS
for ep in range(EPOCHS):
    net.train(); np.random.shuffle(tr); t0=time.time(); tl=0
    for k in range(nb):
        b=tr[k*BS:(k+1)*BS]
        with torch.autocast("cuda",dtype=torch.bfloat16): p=net(tbatch(b))
        if ARCH=="spec":
            y=Zt[b]; e=(p-y).abs()/y.clamp(min=1e-6)
            loss=e[:,METRIC].mean() if METRIC>=0 else e.mean()
        else:
            loss=(p-Zt[b]).abs().mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step(); sched.step(); tl+=loss.item()
    print(f"ep{ep} trloss{tl/nb:.3f} {time.time()-t0:.0f}s",flush=True)
    if (ep+1)%5==0 or ep==EPOCHS-1: torch.save(net.state_dict(),f"{ROOT}/final9000_{TAG}_ep{ep+1}.pt")
torch.save(net.state_dict(),f"{ROOT}/final9000_{TAG}.pt")
print(f"DONE -> final9000_{TAG}.pt",flush=True)
