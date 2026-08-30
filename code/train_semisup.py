"""Semi-supervised multitask fine-tune (user's design: 25 seed + EMA mean-teacher + progressive).
MAIN: regress CD/CV/HEX from 9000 scalars (central-512 cache, the proven 9.4% setup).
AUX (cell-center detection decoder on shared cpsam encoder):
  - SUPERVISED on the 25 green points (256 crop inside their roi_box -> Gaussian heatmap).
  - CONSISTENCY on the pool via EMA teacher (weak-aug teacher heatmap == strong-aug student), ramped.
Goal: force encoder to learn cell-localization features -> hope to lift weak CV/HEX. A/B vs 9.4%.
"""
import os, json, numpy as np, torch, torch.nn as nn, cv2, time, copy, SimpleITK as sitk
from scipy.ndimage import gaussian_filter
from cellpose import models
ROOT="datasets/clear_ec_official"; SZ=512; CROP=256; dev="cuda"
LAMS=float(os.environ.get("LAMS","1.0")); LAMC=float(os.environ.get("LAMC","0.1"))
EPOCHS=int(os.environ.get("EPOCHS","8")); BS=int(os.environ.get("BS","8")); TAG=os.environ.get("TAG","semisup")
meta=np.load(f"{ROOT}/cache_native_meta_{SZ}.npz",allow_pickle=True)
Y=meta["Y"].astype(np.float32); N=int(meta["N"]); ids=meta["ids"]
imgs=np.array(np.memmap(f"{ROOT}/img_native_{SZ}.dat",dtype=np.uint8,mode='r',shape=(N,SZ,SZ)))
rng=np.random.default_rng(42); perm=rng.permutation(N); te,va,tr=perm[:1000],perm[1000:2000],perm[2000:]
ymean=Y[tr].mean(0); Yn=Y/ymean
def norm99(u):
    lo,hi=np.percentile(u,(1,99)); return np.clip((u.astype(np.float32)-lo)/(hi-lo+1e-6),0,1)
IWc,IHc=1296,972
# ---- 25 green detection data (roi_box crops) ----
ann=json.load(open(f"{ROOT}/green_eval/annotations.json"))
GR=[]
for ID in ann:
    u=sitk.GetArrayFromImage(sitk.ReadImage(f"{ROOT}/green_eval/mha/{ID}.mha")).astype(np.float32)
    if u.ndim==3: u=u[0]
    GR.append((u, np.array(ann[ID]["markers_mha"],float), ann[ID]["roi_box_mha"]))
def det_sample():
    u,pts,box=GR[np.random.randint(len(GR))]; x0,y0,x1,y1=[int(v) for v in box]
    cx=np.random.randint(x0,max(x0+1,x1-CROP+1)); cy=np.random.randint(y0,max(y0+1,y1-CROP+1))
    cx=min(cx,IWc-CROP); cy=min(cy,IHc-CROP)
    im=norm99(u[cy:cy+CROP,cx:cx+CROP])
    hm=np.zeros((CROP,CROP),np.float32)
    for px,py in pts:
        xi=int(px-cx); yi=int(py-cy)
        if 0<=xi<CROP and 0<=yi<CROP: hm[yi,xi]=1
    hm=gaussian_filter(hm,3.0); hm=hm/(hm.max()+1e-6)
    if np.random.rand()<.5: im=im[:,::-1]; hm=hm[:,::-1]
    if np.random.rand()<.5: im=im[::-1]; hm=hm[::-1]
    k=np.random.randint(4); im=np.rot90(im,k); hm=np.rot90(hm,k)
    return np.stack([im]*3,0), hm[None]
# ---- pool crops (regression + consistency) ----
def pool_crop(i, strong):
    r,c=np.random.randint(0,SZ-CROP+1),np.random.randint(0,SZ-CROP+1)
    im=norm99(imgs[i][r:r+CROP,c:c+CROP])
    if np.random.rand()<.5: im=im[:,::-1]
    if np.random.rand()<.5: im=im[::-1]
    im=np.rot90(im,np.random.randint(4))
    if strong:   # strong photometric aug for student
        im=np.clip(im*np.random.uniform(0.7,1.3)+np.random.uniform(-0.1,0.1),0,1)
        g=np.random.uniform(0.7,1.4); im=np.clip(im,1e-4,1)**g
        im=np.clip(im+np.random.normal(0,0.03,im.shape),0,1)
    return np.stack([im]*3,0).astype(np.float32)

class Net(nn.Module):
    def __init__(s, enc):
        super().__init__(); s.enc=enc; s.pool=nn.AdaptiveAvgPool2d(1); s.ln=nn.LayerNorm(256)
        def h():
            m=nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Dropout(.2),nn.Linear(128,1))
            nn.init.zeros_(m[-1].weight); nn.init.zeros_(m[-1].bias); return m
        s.hCD,s.hCV,s.hHEX=h(),h(),h()
        def up(i,o): return nn.Sequential(nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),
                     nn.Conv2d(i,o,3,1,1),nn.GroupNorm(8,o),nn.GELU())
        s.dec=nn.Sequential(up(256,128),up(128,64),up(64,32),nn.Conv2d(32,1,1))  # 32->256
    def feat(s,x): return s.enc(x)
    def forward(s,x,det=False):
        f=s.enc(x)
        if det: return s.dec(f)
        g=s.ln(s.pool(f).flatten(1)); return torch.cat([s.hCD(g),s.hCV(g),s.hHEX(g)],1)+1.0
base=models.CellposeModel(gpu=True, pretrained_model="cpsam_models/cpsam_v2")
net=Net(base.net.encoder.float()).to(dev)
import torch.utils.checkpoint as cp
for blk in net.enc.blocks:
    blk._of=blk.forward; blk.forward=(lambda m:(lambda x:cp.checkpoint(m._of,x,use_reentrant=False)))(blk)
ema=copy.deepcopy(net); [p.requires_grad_(False) for p in ema.parameters()]
opt=torch.optim.AdamW([{"params":net.enc.parameters(),"lr":1e-5},
    {"params":list(net.hCD.parameters())+list(net.hCV.parameters())+list(net.hHEX.parameters())+list(net.dec.parameters()),"lr":1e-3}],weight_decay=1e-4)
Ytr=torch.tensor(Yn,device=dev); WARM=200
sched=torch.optim.lr_scheduler.LambdaLR(opt, lambda s:min(1.0,(s+1)/WARM))
def mloss(p,idx): y=Ytr[idx]; return ((p-y).abs()/y.clamp(min=1e-6)).mean()
@torch.no_grad()
def emastep(d=0.99):
    for e,p in zip(ema.parameters(),net.parameters()): e.mul_(d).add_(p,alpha=1-d)
    for e,p in zip(ema.buffers(),net.buffers()): e.copy_(p)
def evalm(idx):
    net.eval(); C5=[(0,0),(0,256),(256,0),(256,256),(128,128)]; es=[]
    with torch.no_grad():
        for s in range(0,len(idx),BS):
            b=idx[s:s+BS]
            x=np.concatenate([np.stack([np.stack([norm99(imgs[i][r:r+CROP,c:c+CROP])]*3,0) for r,c in C5]) for i in b],0)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                p=net(torch.from_numpy(x).to(dev)).float().view(len(b),5,3).mean(1)
            y=Ytr[b]; es.append(((p-y).abs()/y.clamp(min=1e-6)).cpu())
    return torch.cat(es).mean(0).numpy()*100
mse=nn.MSELoss()
best=1e9; bp=f"{ROOT}/regress_ft_best_{TAG}.pt"; nb=len(tr)//BS
step=0
for ep in range(EPOCHS):
    net.train(); np.random.shuffle(tr); t0=time.time(); tl=trg=tds=tcs=0
    ramp=min(1.0, ep/3.0)
    for k in range(nb):
        b=tr[k*BS:(k+1)*BS]
        xw=torch.from_numpy(np.stack([pool_crop(i,False) for i in b])).to(dev)   # weak (reg + teacher)
        xs=torch.from_numpy(np.stack([pool_crop(i,True)  for i in b])).to(dev)   # strong (student cons)
        ds=[det_sample() for _ in range(4)]; xd=torch.from_numpy(np.stack([d[0] for d in ds])).to(dev)
        yd=torch.from_numpy(np.stack([d[1] for d in ds])).to(dev)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            preg=net(xw); Lreg=mloss(preg,b)
            hs=net(xs,det=True)
            with torch.no_grad(): ht=ema(xw,det=True)
            Lcons=mse(hs.float(),ht.float())
            hd=net(xd,det=True); Lsup=mse(hd.float(),yd.float())
        loss=Lreg+LAMS*Lsup+LAMC*ramp*Lcons
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.0)
        opt.step(); sched.step(); emastep(); step+=1
        tl+=loss.item(); trg+=Lreg.item(); tds+=Lsup.item(); tcs+=Lcons.item()
    vm=evalm(va); vov=vm.mean()
    print(f"ep{ep} reg{trg/nb:.3f} sup{tds/nb:.3f} cons{tcs/nb:.3f} | val CD{vm[0]:.1f} CV{vm[1]:.1f} HEX{vm[2]:.1f} OV{vov:.1f} {time.time()-t0:.0f}s",flush=True)
    if vov<best: best=vov; torch.save(net.state_dict(),bp)
net.load_state_dict(torch.load(bp)); tm=evalm(te); vm=evalm(va)
print(f"\n=== SEMISUP FINAL (best val {best:.1f}) ===\nVAL  CD{vm[0]:.1f} CV{vm[1]:.1f} HEX{vm[2]:.1f} OV{vm.mean():.1f}\nTEST CD{tm[0]:.1f} CV{tm[1]:.1f} HEX{tm[2]:.1f} OV{tm.mean():.1f}",flush=True)
json.dump({"test":tm.tolist(),"val":vm.tolist(),"best_val":float(best)},open(f"{ROOT}/regress_ft_results_{TAG}.json","w"),indent=1)
