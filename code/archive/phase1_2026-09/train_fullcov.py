"""Full-image-coverage regression: train on 256 crops sampled from the WHOLE 972x1296 image (not
just central 512), eval by densely tiling the full image (24 tiles) and averaging. GT CV/HEX are
computed over ALL cells, so full coverage reduces the local->global mismatch that may cap CV/HEX.
log-Huber + cosine, mean-pool. Select on val, report test."""
import os, numpy as np, torch, torch.nn as nn, time, math, json
from cellpose import models
ROOT="datasets/clear_ec_official"; H,W=972,1296; CROP=256; dev="cuda"
EPOCHS=int(os.environ.get("EPOCHS","12")); BS=int(os.environ.get("BS","16")); TAG=os.environ.get("TAG","fullcov")
meta=np.load(f"{ROOT}/cache_native_meta_512.npz",allow_pickle=True); Y=meta["Y"].astype(np.float32); N=int(meta["N"])
imgs=np.memmap(f"{ROOT}/img_full_clahe.dat",dtype=np.uint8,mode='r',shape=(N,H,W))  # 11GB, keep on disk
rng=np.random.default_rng(42); perm=rng.permutation(N); te,va,tr=perm[:1000],perm[1000:2000],perm[2000:]
logY=np.log(np.clip(Y,1e-6,None)); lm=logY[tr].mean(0); ls=logY[tr].std(0)
Yln=np.clip((logY-lm)/ls,-4,4).astype(np.float32)
Yln_t=torch.tensor(Yln,device=dev); ls_t=torch.tensor(ls,device=dev); lm_t=torch.tensor(lm,device=dev); Yreal_t=torch.tensor(Y,device=dev)
def norm99(u):
    lo,hi=np.percentile(u,(1,99)); return np.clip((u.astype(np.float32)-lo)/(hi-lo+1e-6),0,1)
RS=[int(x) for x in np.linspace(0,H-CROP,4)]; CS=[int(x) for x in np.linspace(0,W-CROP,6)]  # 24 eval tiles
TILES=[(r,c) for r in RS for c in CS]
class Net(nn.Module):
    def __init__(s,enc):
        super().__init__(); s.enc=enc; s.pool=nn.AdaptiveAvgPool2d(1); s.ln=nn.LayerNorm(256)
        def h(): m=nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Dropout(.2),nn.Linear(128,1)); nn.init.zeros_(m[-1].weight); nn.init.zeros_(m[-1].bias); return m
        s.hCD,s.hCV,s.hHEX=h(),h(),h()
    def forward(s,x): f=s.enc(x); g=s.ln(s.pool(f).flatten(1)); return torch.cat([s.hCD(g),s.hCV(g),s.hHEX(g)],1)
base=models.CellposeModel(gpu=True, pretrained_model="cpsam_models/cpsam_v2")
net=Net(base.net.encoder.float()).to(dev)
import torch.utils.checkpoint as cp
for blk in net.enc.blocks: blk._of=blk.forward; blk.forward=(lambda m:(lambda x:cp.checkpoint(m._of,x,use_reentrant=False)))(blk)
opt=torch.optim.AdamW([{"params":net.enc.parameters(),"lr":1e-5},
    {"params":list(net.hCD.parameters())+list(net.hCV.parameters())+list(net.hHEX.parameters()),"lr":1e-3}],weight_decay=1e-4)
nb=len(tr)//BS; WARM=200; TOTAL=EPOCHS*nb
def lr_lambda(st):
    if st<WARM: return (st+1)/WARM
    p=(st-WARM)/max(1,TOTAL-WARM); return 0.5*(1+math.cos(math.pi*min(1.0,p)))
sched=torch.optim.lr_scheduler.LambdaLR(opt,lr_lambda); huber=nn.SmoothL1Loss(beta=1.0)
def evalm(idx):
    net.eval(); es=[]
    with torch.no_grad():
        for st in range(0,len(idx),4):
            b=idx[st:st+4]; x=np.concatenate([np.stack([np.stack([norm99(imgs[i][r:r+CROP,c:c+CROP])]*3,0) for r,c in TILES]) for i in b],0)
            with torch.autocast("cuda",dtype=torch.bfloat16): p=net(torch.from_numpy(x).to(dev)).float().view(len(b),len(TILES),3).mean(1)
            yhat=torch.exp(p*ls_t+lm_t); y=Yreal_t[b]; es.append(((yhat-y).abs()/y).cpu())
    return torch.cat(es).mean(0).numpy()*100
best=1e9; bp=f"{ROOT}/regress_ft_best_{TAG}.pt"
for ep in range(EPOCHS):
    net.train(); np.random.shuffle(tr); t0=time.time(); trg=0
    for k in range(nb):
        b=tr[k*BS:(k+1)*BS]
        xb=np.stack([np.stack([norm99(imgs[i][r:r+CROP,c:c+CROP])]*3,0) for i in b
            for r,c in [(np.random.randint(0,H-CROP+1),np.random.randint(0,W-CROP+1))]])
        with torch.autocast("cuda",dtype=torch.bfloat16):
            preg=net(torch.from_numpy(xb).to(dev)); loss=huber(preg,Yln_t[b])
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step(); sched.step(); trg+=loss.item()
    vm=evalm(va); vov=vm.mean()
    print(f"ep{ep} reg{trg/nb:.3f} | val CD{vm[0]:.1f} CV{vm[1]:.1f} HEX{vm[2]:.1f} OV{vov:.2f} {time.time()-t0:.0f}s",flush=True)
    if vov<best: best=vov; torch.save(net.state_dict(),bp)
net.load_state_dict(torch.load(bp)); tm=evalm(te); vm=evalm(va)
print(f"\n=== {TAG} FINAL (best val {best:.2f}) ===\nVAL  CD{vm[0]:.1f} CV{vm[1]:.1f} HEX{vm[2]:.1f} OV{vm.mean():.2f}\nTEST CD{tm[0]:.1f} CV{tm[1]:.1f} HEX{tm[2]:.1f} OV{tm.mean():.2f}",flush=True)
json.dump({"test":tm.tolist(),"val":vm.tolist(),"best_val":float(best)},open(f"{ROOT}/regress_ft_results_{TAG}.json","w"),indent=1)
