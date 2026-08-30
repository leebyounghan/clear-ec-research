"""SSL point detector: train on ~1500 tr POOL images with fNLA-mask-centroid pseudo point-labels
(fnla_points_pool.npz) instead of only 25 green -> fixes the val under-detection that made point-CD
19%. cpsam enc + heatmap dec, MSE to Gaussian centers. Full CLAHE images (already CLAHE -> percentile
norm only, matching inference prep). Saves pointdet_ssl.pt. No val/test rows (labels are tr-only)."""
import os, numpy as np, torch, torch.nn as nn, cv2, time
from scipy.ndimage import gaussian_filter
from cellpose import models
ROOT="datasets/clear_ec_official"; dev="cuda"; CROP=256; H,W=972,1296
EPOCHS=int(os.environ.get("EPOCHS","30")); BS=int(os.environ.get("BS","8")); SIG=float(os.environ.get("SIG","3.0")); SPE=int(os.environ.get("SPE","90"))
Z=np.load(f"{ROOT}/fnla_points_pool.npz"); rows=list(Z.keys()); PTS={r:Z[r] for r in rows}
cm=np.load(f"{ROOT}/cache_native_meta_512.npz",allow_pickle=True); Nc=int(cm["N"])
full=np.memmap(f"{ROOT}/img_full_clahe.dat",dtype=np.uint8,mode='r',shape=(Nc,H,W))
print(f"SSL pointdet: {len(rows)} tr images, ~{np.mean([len(v) for v in PTS.values()]):.0f} pts/img",flush=True)
def norm(u):  # u already CLAHE (full cache) -> percentile-normalize (matches inference after its CLAHE)
    lo,hi=np.percentile(u,(1,99)); return np.clip((u.astype(np.float32)-lo)/(hi-lo+1e-6),0,1)
def sample():
    r=rows[np.random.randint(len(rows))]; img=np.asarray(full[int(r)]); pts=PTS[r]
    cy=np.random.randint(0,H-CROP); cx=np.random.randint(0,W-CROP)
    im=norm(img[cy:cy+CROP,cx:cx+CROP]); hm=np.zeros((CROP,CROP),np.float32)
    for px,py in pts:
        xi=int(px-cx); yi=int(py-cy)
        if 0<=xi<CROP and 0<=yi<CROP: hm[yi,xi]=1
    hm=gaussian_filter(hm,SIG); hm=hm/(hm.max()+1e-6)
    im=np.clip(im*np.random.uniform(0.7,1.3)+np.random.uniform(-0.1,0.1),0,1)
    im=np.clip(im,1e-4,1)**np.random.uniform(0.7,1.4)
    im=np.clip(im+np.random.normal(0,0.03,im.shape),0,1).astype(np.float32)
    if np.random.rand()<.5: im=im[:,::-1]; hm=hm[:,::-1]
    if np.random.rand()<.5: im=im[::-1]; hm=hm[::-1]
    k=np.random.randint(4); im=np.rot90(im,k); hm=np.rot90(hm,k)
    return np.stack([im]*3,0).copy(), hm[None].copy()
class Det(nn.Module):
    def __init__(s,enc):
        super().__init__(); s.enc=enc
        def up(i,o): return nn.Sequential(nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),nn.Conv2d(i,o,3,1,1),nn.GroupNorm(8,o),nn.GELU())
        s.dec=nn.Sequential(up(256,128),up(128,64),up(64,32),nn.Conv2d(32,1,1))
    def forward(s,x): return s.dec(s.enc(x))
base=models.CellposeModel(gpu=True, pretrained_model="cpsam_models/cpsam_v2")
net=Det(base.net.encoder.float()).to(dev)
# warm-start from the green-trained detector
try: net.load_state_dict(torch.load(f"{ROOT}/pointdet.pt")); print("warm-started from pointdet.pt",flush=True)
except Exception as e: print("fresh init",e,flush=True)
import torch.utils.checkpoint as cp
for blk in net.enc.blocks:
    blk._of=blk.forward; blk.forward=(lambda m:(lambda x:cp.checkpoint(m._of,x,use_reentrant=False)))(blk)
opt=torch.optim.AdamW([{"params":net.enc.parameters(),"lr":1e-5},{"params":net.dec.parameters(),"lr":1e-3}],weight_decay=1e-4)
mse=nn.MSELoss()
for ep in range(EPOCHS):
    net.train(); tl=0; t0=time.time()
    for k in range(SPE):
        xb=[];yb=[]
        for _ in range(BS): a,b=sample(); xb.append(a); yb.append(b)
        x=torch.from_numpy(np.stack(xb)).to(dev); y=torch.from_numpy(np.stack(yb)).to(dev)
        with torch.autocast("cuda",dtype=torch.bfloat16): p=net(x); loss=mse(p.float(),y)
        opt.zero_grad(); loss.backward(); opt.step(); tl+=loss.item()
    print(f"ep{ep} loss{tl/SPE:.5f} {time.time()-t0:.0f}s",flush=True)
torch.save(net.state_dict(), f"{ROOT}/pointdet_ssl.pt"); print(f"DONE -> {ROOT}/pointdet_ssl.pt",flush=True)
