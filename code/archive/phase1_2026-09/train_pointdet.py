"""Cell-center point detector: cpsam encoder + upsample decoder -> Gaussian center heatmap.
Trained on the 25 green (GT points), heavy aug. Deployed to detect points on the 9000 so the
SAM point-prompt pipeline can generate DIVERSE pseudo-masks (fixes cpsam nc=0 generalization).
256 input (encoder pos_embed fixed at 32x32). Saves pointdet.pt."""
import os, json, numpy as np, torch, torch.nn as nn, cv2, time, SimpleITK as sitk
from scipy.ndimage import gaussian_filter
from cellpose import models
ROOT="datasets/clear_ec_official"; OUT=f"{ROOT}/green_eval"; dev="cuda"; CROP=256
EPOCHS=int(os.environ.get("EPOCHS","40")); BS=int(os.environ.get("BS","8")); SIG=float(os.environ.get("SIG","3.0"))
_cl=cv2.createCLAHE(2.0,(8,8))
def norm(u):
    u=_cl.apply(np.clip(u,0,255).astype(np.uint8)).astype(np.float32)
    lo,hi=np.percentile(u,(1,99)); return np.clip((u-lo)/(hi-lo+1e-6),0,1)
ann=json.load(open(f"{OUT}/annotations.json")); GR=[]
for ID in ann:
    u=sitk.GetArrayFromImage(sitk.ReadImage(f"{OUT}/mha/{ID}.mha")).astype(np.float32)
    if u.ndim==3: u=u[0]
    GR.append((u,np.array(ann[ID]["markers_mha"],float),ann[ID]["roi_box_mha"]))
IWc,IHc=1296,972
def sample():
    u,pts,box=GR[np.random.randint(len(GR))]; H,W=u.shape
    cy=np.random.randint(0,max(1,H-CROP)); cx=np.random.randint(0,max(1,W-CROP))
    im=norm(u[cy:cy+CROP,cx:cx+CROP]); hm=np.zeros((CROP,CROP),np.float32)
    for px,py in pts:
        xi=int(px-cx); yi=int(py-cy)
        if 0<=xi<CROP and 0<=yi<CROP: hm[yi,xi]=1
    hm=gaussian_filter(hm,SIG); hm=hm/(hm.max()+1e-6)
    # photometric + geometric aug
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
        def up(i,o): return nn.Sequential(nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),
                     nn.Conv2d(i,o,3,1,1),nn.GroupNorm(8,o),nn.GELU())
        s.dec=nn.Sequential(up(256,128),up(128,64),up(64,32),nn.Conv2d(32,1,1))
    def forward(s,x): return s.dec(s.enc(x))
base=models.CellposeModel(gpu=True, pretrained_model="cpsam_models/cpsam_v2")
net=Det(base.net.encoder.float()).to(dev)
import torch.utils.checkpoint as cp
for blk in net.enc.blocks:
    blk._of=blk.forward; blk.forward=(lambda m:(lambda x:cp.checkpoint(m._of,x,use_reentrant=False)))(blk)
opt=torch.optim.AdamW([{"params":net.enc.parameters(),"lr":1e-5},{"params":net.dec.parameters(),"lr":1e-3}],weight_decay=1e-4)
mse=nn.MSELoss(); SPE=int(os.environ.get("SPE","60"))
for ep in range(EPOCHS):
    net.train(); tl=0; t0=time.time()
    for k in range(SPE):
        xb=[];yb=[]
        for _ in range(BS): a,b=sample(); xb.append(a); yb.append(b)
        x=torch.from_numpy(np.stack(xb)).to(dev); y=torch.from_numpy(np.stack(yb)).to(dev)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            p=net(x); loss=mse(p.float(),y)
        opt.zero_grad(); loss.backward(); opt.step(); tl+=loss.item()
    print(f"ep{ep} loss{tl/SPE:.5f} {time.time()-t0:.0f}s",flush=True)
torch.save(net.state_dict(), f"{ROOT}/pointdet.pt"); print(f"DONE -> {ROOT}/pointdet.pt",flush=True)
