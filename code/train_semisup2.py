"""Semi-supervised multitask v2 — FIXED + IMPROVED.
Fixes vs v1: (1) LOG-space targets + Huber (SmoothL1) loss -> robust to tiny-GT outliers, kills
oscillation, handles CV/HEX multiplicative range. (2) EMA consistency on ALIGNED crops (same crop +
geometry, only photometric differs) -> proper mean-teacher. Aux = cell-center detection seeded by 25.
Eval converts log-pred back to real units -> challenge MAPE. A/B vs baseline 9.4%."""
import os, json, numpy as np, torch, torch.nn as nn, cv2, time, copy, SimpleITK as sitk
from scipy.ndimage import gaussian_filter
from cellpose import models
ROOT="datasets/clear_ec_official"; SZ=512; CROP=int(os.environ.get("CROP","256")); dev="cuda"
LAMS=float(os.environ.get("LAMS","1.0")); LAMC=float(os.environ.get("LAMC","0.1"))
EPOCHS=int(os.environ.get("EPOCHS","6")); BS=int(os.environ.get("BS","8")); TAG=os.environ.get("TAG","semisup2")
USE_AUX=int(os.environ.get("USE_AUX","1")); CLAHE=int(os.environ.get("CLAHE","0")); _cl=cv2.createCLAHE(2.0,(8,8))
meta=np.load(f"{ROOT}/cache_native_meta_{SZ}.npz",allow_pickle=True)
Y=meta["Y"].astype(np.float32); N=int(meta["N"])
_srcf="img_clahe" if os.environ.get("CLAHE_SRC") else "img_native"
imgs=np.array(np.memmap(f"{ROOT}/{_srcf}_{SZ}.dat",dtype=np.uint8,mode='r',shape=(N,SZ,SZ)))
if os.environ.get("CLAHE_SRC"): CLAHE=0  # already CLAHE'd; norm99 just percentile-normalizes
rng=np.random.default_rng(42); perm=rng.permutation(N); te,va,tr=perm[:1000],perm[1000:2000],perm[2000:]
# ---- log-space normalized targets (robust) ----
logY=np.log(np.clip(Y,1e-6,None)); lm=logY[tr].mean(0); ls=logY[tr].std(0)
Yln=np.clip((logY-lm)/ls,-4,4).astype(np.float32)
Yln_t=torch.tensor(Yln,device=dev); ls_t=torch.tensor(ls,device=dev); lm_t=torch.tensor(lm,device=dev)
Yreal_t=torch.tensor(Y,device=dev)
IWc,IHc=1296,972
def norm99(u):
    if CLAHE: u=_cl.apply(np.clip(u,0,255).astype(np.uint8)).astype(np.float32)
    lo,hi=np.percentile(u,(1,99)); return np.clip((u.astype(np.float32)-lo)/(hi-lo+1e-6),0,1)
# ---- 25 green detection data ----
ann=json.load(open(f"{ROOT}/green_eval/annotations.json")); GR=[]
for ID in ann:
    u=sitk.GetArrayFromImage(sitk.ReadImage(f"{ROOT}/green_eval/mha/{ID}.mha")).astype(np.float32)
    if u.ndim==3: u=u[0]
    GR.append((u,np.array(ann[ID]["markers_mha"],float),ann[ID]["roi_box_mha"]))
def det_sample():
    u,pts,box=GR[np.random.randint(len(GR))]; x0,y0,x1,y1=[int(v) for v in box]
    cx=min(np.random.randint(x0,max(x0+1,x1-CROP+1)),IWc-CROP); cy=min(np.random.randint(y0,max(y0+1,y1-CROP+1)),IHc-CROP)
    im=norm99(u[cy:cy+CROP,cx:cx+CROP]); hm=np.zeros((CROP,CROP),np.float32)
    for px,py in pts:
        xi=int(px-cx); yi=int(py-cy)
        if 0<=xi<CROP and 0<=yi<CROP: hm[yi,xi]=1
    hm=gaussian_filter(hm,3.0); hm=hm/(hm.max()+1e-6)
    if np.random.rand()<.5: im=im[:,::-1]; hm=hm[:,::-1]
    if np.random.rand()<.5: im=im[::-1]; hm=hm[::-1]
    k=np.random.randint(4); im=np.rot90(im,k); hm=np.rot90(hm,k)
    return np.stack([im]*3,0).copy(), hm[None].copy()
def photo(im):
    im=np.clip(im*np.random.uniform(0.7,1.3)+np.random.uniform(-0.1,0.1),0,1)
    g=np.random.uniform(0.7,1.4); im=np.clip(im,1e-4,1)**g
    return np.clip(im+np.random.normal(0,0.03,im.shape),0,1).astype(np.float32)
def pool_pair(i):
    """one crop + shared geometry; weak(reg/teacher) and strong(student) differ only in photometry."""
    r,c=np.random.randint(0,SZ-CROP+1),np.random.randint(0,SZ-CROP+1); im=norm99(imgs[i][r:r+CROP,c:c+CROP])
    if np.random.rand()<.5: im=im[:,::-1]
    if np.random.rand()<.5: im=im[::-1]
    im=np.rot90(im,np.random.randint(4)).copy()
    weak=np.stack([im]*3,0).astype(np.float32); strong=np.stack([photo(im)]*3,0)
    return weak, strong

class Net(nn.Module):
    def __init__(s, enc):
        super().__init__(); s.enc=enc; s.pool=nn.AdaptiveAvgPool2d(1); s.ln=nn.LayerNorm(256)
        s.POOL=os.environ.get("POOL","mean")
        if s.POOL=="attn":
            s.qtok=nn.Parameter(torch.zeros(1,1,256)); nn.init.normal_(s.qtok,std=.02)
            s.attn=nn.MultiheadAttention(256,4,batch_first=True)
        def h():
            m=nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Dropout(.2),nn.Linear(128,1))
            nn.init.zeros_(m[-1].weight); nn.init.zeros_(m[-1].bias); return m
        s.hCD,s.hCV,s.hHEX=h(),h(),h()
        def up(i,o): return nn.Sequential(nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),
                     nn.Conv2d(i,o,3,1,1),nn.GroupNorm(8,o),nn.GELU())
        s.dec=nn.Sequential(up(256,128),up(128,64),up(64,32),nn.Conv2d(32,1,1))
    def forward(s,x,det=False):
        f=s.enc(x)
        if det: return s.dec(f)
        if s.POOL=="attn":
            B=f.shape[0]; seq=f.flatten(2).transpose(1,2)   # (B,1024,256)
            g=s.attn(s.qtok.expand(B,-1,-1),seq,seq)[0].squeeze(1)
        else: g=s.pool(f).flatten(1)
        g=s.ln(g); return torch.cat([s.hCD(g),s.hCV(g),s.hHEX(g)],1)   # NO +1 offset (log-mean=0)
base=models.CellposeModel(gpu=True, pretrained_model="cpsam_models/cpsam_v2")
net=Net(base.net.encoder.float()).to(dev)
if os.environ.get("MAE_INIT"):
    net.enc.load_state_dict(torch.load(f"{ROOT}/mae_encoder.pt")); print("loaded MAE-pretrained encoder",flush=True)
import torch.utils.checkpoint as cp
for blk in net.enc.blocks:
    blk._of=blk.forward; blk.forward=(lambda m:(lambda x:cp.checkpoint(m._of,x,use_reentrant=False)))(blk)
ema=copy.deepcopy(net); [p.requires_grad_(False) for p in ema.parameters()]
opt=torch.optim.AdamW([{"params":net.enc.parameters(),"lr":1e-5},
    {"params":list(net.hCD.parameters())+list(net.hCV.parameters())+list(net.hHEX.parameters())+list(net.dec.parameters()),"lr":1e-3}],weight_decay=1e-4)
WARM=200; nb=len(tr)//BS; TOTAL=EPOCHS*nb
import math
def lr_lambda(s):
    if s<WARM: return (s+1)/WARM
    prog=(s-WARM)/max(1,TOTAL-WARM); return 0.5*(1+math.cos(math.pi*min(1.0,prog)))
sched=torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
huber=nn.SmoothL1Loss(beta=1.0); mse=nn.MSELoss()
huber_nr=nn.SmoothL1Loss(beta=1.0,reduction='none')
WT=os.environ.get("WT","")  # per-metric loss weights "wCD,wCV,wHEX" (default equal)
wt_t=torch.tensor([float(x) for x in WT.split(",")],device=dev) if WT else None
@torch.no_grad()
def emastep(d=0.99):
    for e,p in zip(ema.parameters(),net.parameters()): e.mul_(d).add_(p,alpha=1-d)
    for e,p in zip(ema.buffers(),net.buffers()): e.copy_(p)
def evalm(idx):
    net.eval(); m=SZ-CROP
    C5=[(0,0)] if m==0 else [(0,0),(0,m),(m,0),(m,m),(m//2,m//2)]; nc=len(C5); es=[]
    with torch.no_grad():
        for s in range(0,len(idx),BS):
            b=idx[s:s+BS]
            x=np.concatenate([np.stack([np.stack([norm99(imgs[i][r:r+CROP,c:c+CROP])]*3,0) for r,c in C5]) for i in b],0)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                p=net(torch.from_numpy(x).to(dev)).float().view(len(b),nc,3).mean(1)
            yhat=torch.exp(p*ls_t+lm_t); y=Yreal_t[b]; es.append(((yhat-y).abs()/y).cpu())
    return torch.cat(es).mean(0).numpy()*100
best=1e9; bp=f"{ROOT}/regress_ft_best_{TAG}.pt"; nb=len(tr)//BS
for ep in range(EPOCHS):
    net.train(); np.random.shuffle(tr); t0=time.time(); tl=trg=tds=tcs=0; ramp=min(1.0,ep/2.0)
    for k in range(nb):
        b=tr[k*BS:(k+1)*BS]
        pairs=[pool_pair(i) for i in b]
        xw=torch.from_numpy(np.stack([p[0] for p in pairs])).to(dev)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            preg=net(xw)
            if wt_t is not None: Lreg=(huber_nr(preg, Yln_t[b])*wt_t).mean()
            else: Lreg=huber(preg, Yln_t[b])
            loss=Lreg
            if USE_AUX:
                xs=torch.from_numpy(np.stack([p[1] for p in pairs])).to(dev)
                hs=net(xs,det=True)
                with torch.no_grad(): ht=ema(xw,det=True)
                Lcons=mse(hs.float(),ht.float())
                ds=[det_sample() for _ in range(4)]
                xd=torch.from_numpy(np.stack([d[0] for d in ds])).to(dev); yd=torch.from_numpy(np.stack([d[1] for d in ds])).to(dev)
                hd=net(xd,det=True); Lsup=mse(hd.float(),yd.float())
                loss=Lreg+LAMS*Lsup+LAMC*ramp*Lcons; tds+=Lsup.item(); tcs+=Lcons.item()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.0)
        opt.step(); sched.step()
        if USE_AUX: emastep()
        tl+=loss.item(); trg+=Lreg.item()
    vm=evalm(va); vov=vm.mean()
    print(f"ep{ep} reg{trg/nb:.3f} sup{tds/nb:.3f} cons{tcs/nb:.3f} | val CD{vm[0]:.1f} CV{vm[1]:.1f} HEX{vm[2]:.1f} OV{vov:.2f} {time.time()-t0:.0f}s",flush=True)
    if vov<best: best=vov; torch.save(net.state_dict(),bp)
net.load_state_dict(torch.load(bp)); tm=evalm(te); vm=evalm(va)
print(f"\n=== {TAG} FINAL (best val {best:.2f}) ===\nVAL  CD{vm[0]:.1f} CV{vm[1]:.1f} HEX{vm[2]:.1f} OV{vm.mean():.2f}\nTEST CD{tm[0]:.1f} CV{tm[1]:.1f} HEX{tm[2]:.1f} OV{tm.mean():.2f}",flush=True)
json.dump({"test":tm.tolist(),"val":vm.tolist(),"best_val":float(best)},open(f"{ROOT}/regress_ft_results_{TAG}.json","w"),indent=1)
