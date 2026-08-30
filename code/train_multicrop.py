"""Run C: multi-crop FEATURE aggregation. CV/HEX are population statistics over many cells;
one 256 crop (~60 cells) gives a noisy estimate. Encode K crops per image, mean-pool their
features BEFORE the heads -> the heads see a richer cell population -> hope to lift CV/HEX.
Log-Huber loss (from v2). Eval aggregates K=5 fixed crops at feature level. A/B vs Run A 9.34%."""
import os, json, numpy as np, torch, torch.nn as nn, time
from cellpose import models
ROOT="datasets/clear_ec_official"; SZ=512; CROP=256; dev="cuda"
EPOCHS=int(os.environ.get("EPOCHS","8")); BS=int(os.environ.get("BS","4")); K=int(os.environ.get("K","4")); TAG=os.environ.get("TAG","multicrop")
meta=np.load(f"{ROOT}/cache_native_meta_{SZ}.npz",allow_pickle=True)
Y=meta["Y"].astype(np.float32); N=int(meta["N"])
imgs=np.array(np.memmap(f"{ROOT}/img_native_{SZ}.dat",dtype=np.uint8,mode='r',shape=(N,SZ,SZ)))
rng=np.random.default_rng(42); perm=rng.permutation(N); te,va,tr=perm[:1000],perm[1000:2000],perm[2000:]
logY=np.log(np.clip(Y,1e-6,None)); lm=logY[tr].mean(0); ls=logY[tr].std(0)
Yln=np.clip((logY-lm)/ls,-4,4).astype(np.float32)
Yln_t=torch.tensor(Yln,device=dev); ls_t=torch.tensor(ls,device=dev); lm_t=torch.tensor(lm,device=dev); Yreal_t=torch.tensor(Y,device=dev)
def norm99(u):
    lo,hi=np.percentile(u,(1,99)); return np.clip((u.astype(np.float32)-lo)/(hi-lo+1e-6),0,1)
def one_crop(i):
    r,c=np.random.randint(0,SZ-CROP+1),np.random.randint(0,SZ-CROP+1); im=norm99(imgs[i][r:r+CROP,c:c+CROP])
    if np.random.rand()<.5: im=im[:,::-1]
    if np.random.rand()<.5: im=im[::-1]
    im=np.rot90(im,np.random.randint(4))
    return np.stack([im]*3,0)
def make_multi(i): return np.stack([one_crop(i) for _ in range(K)])   # (K,3,256,256)
C5=[(0,0),(0,256),(256,0),(256,256),(128,128)]
def make_eval(i): return np.stack([np.stack([norm99(imgs[i][r:r+CROP,c:c+CROP])]*3,0) for r,c in C5])  # (5,...)

class Net(nn.Module):
    def __init__(s, enc):
        super().__init__(); s.enc=enc; s.pool=nn.AdaptiveAvgPool2d(1); s.ln=nn.LayerNorm(256)
        def h():
            m=nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Dropout(.2),nn.Linear(128,1))
            nn.init.zeros_(m[-1].weight); nn.init.zeros_(m[-1].bias); return m
        s.hCD,s.hCV,s.hHEX=h(),h(),h()
    def forward(s,x):   # x: (B,k,3,256,256)
        B,k=x.shape[:2]; f=s.enc(x.reshape(B*k,*x.shape[2:])); g=s.pool(f).flatten(1).view(B,k,256).mean(1)
        g=s.ln(g); return torch.cat([s.hCD(g),s.hCV(g),s.hHEX(g)],1)
base=models.CellposeModel(gpu=True, pretrained_model="cpsam_models/cpsam_v2")
net=Net(base.net.encoder.float()).to(dev)
import torch.utils.checkpoint as cp
for blk in net.enc.blocks:
    blk._of=blk.forward; blk.forward=(lambda m:(lambda x:cp.checkpoint(m._of,x,use_reentrant=False)))(blk)
opt=torch.optim.AdamW([{"params":net.enc.parameters(),"lr":1e-5},
    {"params":list(net.hCD.parameters())+list(net.hCV.parameters())+list(net.hHEX.parameters()),"lr":1e-3}],weight_decay=1e-4)
WARM=200; sched=torch.optim.lr_scheduler.LambdaLR(opt, lambda s:min(1.0,(s+1)/WARM)); huber=nn.SmoothL1Loss(beta=1.0)
def evalm(idx):
    net.eval(); es=[]
    with torch.no_grad():
        for s in range(0,len(idx),BS):
            b=idx[s:s+BS]; x=torch.from_numpy(np.stack([make_eval(i) for i in b])).to(dev)
            with torch.autocast("cuda",dtype=torch.bfloat16): p=net(x).float()
            yhat=torch.exp(p*ls_t+lm_t); y=Yreal_t[b]; es.append(((yhat-y).abs()/y).cpu())
    return torch.cat(es).mean(0).numpy()*100
best=1e9; bp=f"{ROOT}/regress_ft_best_{TAG}.pt"; nb=len(tr)//BS
for ep in range(EPOCHS):
    net.train(); np.random.shuffle(tr); t0=time.time(); tl=0
    for kk in range(nb):
        b=tr[kk*BS:(kk+1)*BS]; x=torch.from_numpy(np.stack([make_multi(i) for i in b])).to(dev)
        with torch.autocast("cuda",dtype=torch.bfloat16): p=net(x); loss=huber(p,Yln_t[b])
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step(); sched.step(); tl+=loss.item()
    vm=evalm(va); vov=vm.mean()
    print(f"ep{ep} loss{tl/nb:.3f} | val CD{vm[0]:.1f} CV{vm[1]:.1f} HEX{vm[2]:.1f} OV{vov:.2f} {time.time()-t0:.0f}s",flush=True)
    if vov<best: best=vov; torch.save(net.state_dict(),bp)
net.load_state_dict(torch.load(bp)); tm=evalm(te); vm=evalm(va)
print(f"\n=== {TAG} FINAL (best val {best:.2f}) ===\nVAL  CD{vm[0]:.1f} CV{vm[1]:.1f} HEX{vm[2]:.1f} OV{vm.mean():.2f}\nTEST CD{tm[0]:.1f} CV{tm[1]:.1f} HEX{tm[2]:.1f} OV{tm.mean():.2f}",flush=True)
json.dump({"test":tm.tolist(),"val":vm.tolist(),"best_val":float(best),"K":K},open(f"{ROOT}/regress_ft_results_{TAG}.json","w"),indent=1)
