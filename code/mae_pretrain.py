"""MAE-lite self-supervised pretraining of the cpsam ViT encoder on the 9000 (unlabeled).
Mask random 16x16 patches of the CLAHE-normalized 256 crop; encoder->conv decoder reconstructs the
full grayscale; MSE on masked regions. Saves the pretrained encoder for downstream regression FT.
No labels used -> non-circular. (Research: low upside, but the user's named SSL lever.)"""
import os, numpy as np, torch, torch.nn as nn, cv2, time
from cellpose import models
ROOT="datasets/clear_ec_official"; SZ=512; CROP=256; dev="cuda"
EPOCHS=int(os.environ.get("EPOCHS","3")); BS=int(os.environ.get("BS","16")); MASK=float(os.environ.get("MASK","0.5"))
meta=np.load(f"{ROOT}/cache_native_meta_{SZ}.npz",allow_pickle=True); N=int(meta["N"])
imgs=np.array(np.memmap(f"{ROOT}/img_native_{SZ}.dat",dtype=np.uint8,mode='r',shape=(N,SZ,SZ)))
rng=np.random.default_rng(42); perm=rng.permutation(N); tr=perm[2000:]  # use train split only (no val/test)
_cl=cv2.createCLAHE(2.0,(8,8))
def norm99(u):
    u=_cl.apply(np.clip(u,0,255).astype(np.uint8)).astype(np.float32)
    lo,hi=np.percentile(u,(1,99)); return np.clip((u-lo)/(hi-lo+1e-6),0,1)
def crop(i):
    r,c=np.random.randint(0,SZ-CROP+1),np.random.randint(0,SZ-CROP+1); im=norm99(imgs[i][r:r+CROP,c:c+CROP])
    if np.random.rand()<.5: im=im[:,::-1]
    if np.random.rand()<.5: im=im[::-1]
    im=np.rot90(im,np.random.randint(4)); return im.copy()
P=16; NP=CROP//P  # 16x16 patches -> 16x16 grid
def mask_grid():
    m=(np.random.rand(NP,NP)<MASK).astype(np.float32)
    return np.repeat(np.repeat(m,P,0),P,1)  # (256,256) 1=masked
class Dec(nn.Module):
    def __init__(s):
        super().__init__()
        def up(i,o): return nn.Sequential(nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),nn.Conv2d(i,o,3,1,1),nn.GroupNorm(8,o),nn.GELU())
        s.net=nn.Sequential(up(256,128),up(128,64),up(64,32),nn.Conv2d(32,1,1))   # 32->256, 1ch
    def forward(s,x): return s.net(x)
base=models.CellposeModel(gpu=True, pretrained_model="cpsam_models/cpsam_v2")
enc=base.net.encoder.float().to(dev); dec=Dec().to(dev)
import torch.utils.checkpoint as cp
for blk in enc.blocks:
    blk._of=blk.forward; blk.forward=(lambda m:(lambda x:cp.checkpoint(m._of,x,use_reentrant=False)))(blk)
opt=torch.optim.AdamW(list(enc.parameters())+list(dec.parameters()),lr=1e-5,weight_decay=1e-4)
mse=nn.MSELoss(reduction='none')
nb=len(tr)//BS
print(f"MAE pretrain: {len(tr)} imgs, {EPOCHS}ep, mask{MASK}",flush=True)
for ep in range(EPOCHS):
    np.random.shuffle(tr); t0=time.time(); tl=0
    for k in range(nb):
        b=tr[k*BS:(k+1)*BS]
        ims=np.stack([crop(i) for i in b]); msk=np.stack([mask_grid() for _ in b])   # (B,256,256)
        tgt=torch.from_numpy(ims[:,None]).to(dev)                                     # (B,1,256,256) clean target
        mk=torch.from_numpy(msk[:,None]).to(dev)
        inp=torch.from_numpy((ims*(1-msk))[:,None]).repeat(1,3,1,1).to(dev)            # masked, 3ch
        with torch.autocast("cuda",dtype=torch.bfloat16):
            rec=dec(enc(inp)).float()
            loss=(mse(rec,tgt)*mk).sum()/(mk.sum()+1e-6)                              # MSE on masked only
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(list(enc.parameters())+list(dec.parameters()),1.0)
        opt.step(); tl+=loss.item()
        if k%400==0: print(f"ep{ep} step{k}/{nb} loss{tl/(k+1):.4f} {time.time()-t0:.0f}s",flush=True)
    print(f"ep{ep} done loss{tl/nb:.4f} {time.time()-t0:.0f}s",flush=True)
    # strip checkpoint wrappers before saving
torch.save(enc.state_dict(), f"{ROOT}/mae_encoder.pt")
print("saved mae_encoder.pt",flush=True)
