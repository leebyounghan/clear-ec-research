"""Full-frame regression: same recipe as the best member (clahe_s2: cpsam encoder, per-crop CLAHE,
log-space targets) but crops sampled from the ENTIRE 972x1296 frame instead of the central 512.
Rationale (roi_region_diag): GT is computed on an annotator-chosen patch anywhere in the frame;
CV varies 32% across windows, so the outer 74% of the frame that our models never saw carries the
region signal. Eval: mean over a 4x6 grid of 256-crops (bf16, no dihedral during epochs; final adds
center-512 C5 for comparability). Val-selected; compare vs central-512 clahe_s2 (val 6.4/10.4/10.1)."""
import os, numpy as np, torch, torch.nn as nn, json, time, cv2
from cellpose import models
ROOT="datasets/clear_ec_official"; H,W=972,1296; CROP=256
EPOCHS=int(os.environ.get("EPOCHS","25")); BS=int(os.environ.get("BS","16")); TAG=os.environ.get("TAG","fullframe")
LM=np.array([7.919919967651367,-0.9824771285057068,-0.6211431622505188],np.float32)
LS=np.array([0.18971863389015198,0.18813571333885193,0.2748945653438568],np.float32)
_clahe=cv2.createCLAHE(2.0,(8,8)); dev="cuda"
meta=np.load(f"{ROOT}/cache_native_meta_512.npz",allow_pickle=True)
Y=meta["Y"].astype(np.float32); N=int(meta["N"])
Z=(np.log(np.clip(Y,1e-6,None))-LM)/LS
imgs=np.memmap(f"{ROOT}/img_full_raw.dat",dtype=np.uint8,mode='r',shape=(N,H,W))
rng=np.random.default_rng(42); perm=rng.permutation(N)
te,va,tr=perm[:1000],perm[1000:2000],perm[2000:]
bad=set(np.where((Y<=0).any(1))[0].tolist())      # HEX=0 rows: log target undefined
tr=np.array([i for i in tr if i not in bad]); va=np.array([i for i in va if i not in bad]); te=np.array([i for i in te if i not in bad])
print(f"dropped {len(bad)} zero-GT rows from splits",flush=True)
print(f"split train {len(tr)} val {len(va)} test {len(te)}",flush=True)

def norm99(u):
    u=_clahe.apply(np.ascontiguousarray(u))
    lo,hi=np.percentile(u,(1,99)); return np.clip((u.astype(np.float32)-lo)/(hi-lo+1e-6),0,1)
GRID=[(r,c) for r in np.linspace(0,H-CROP,4).astype(int) for c in np.linspace(0,W-CROP,6).astype(int)]
def make(i, train):
    if train:
        r,c=np.random.randint(0,H-CROP+1),np.random.randint(0,W-CROP+1)
        im=norm99(imgs[i][r:r+CROP,c:c+CROP])
        if np.random.rand()<.5: im=im[:,::-1]
        if np.random.rand()<.5: im=im[::-1,:]
        im=np.ascontiguousarray(np.rot90(im,np.random.randint(4)))
        return np.stack([im]*3,0)[None]
    return np.stack([np.stack([norm99(imgs[i][r:r+CROP,c:c+CROP])]*3,0) for r,c in GRID])
def tbatch(idx, train):
    return torch.from_numpy(np.concatenate([make(i,train) for i in idx],0)).to(dev)

class Net(nn.Module):
    def __init__(s, enc):
        super().__init__(); s.enc=enc; s.pool=nn.AdaptiveAvgPool2d(1); s.ln=nn.LayerNorm(256)
        def head():
            h=nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Dropout(.2),nn.Linear(128,1))
            nn.init.zeros_(h[-1].weight); nn.init.zeros_(h[-1].bias)   # start pred z=0 (= train mean)
            return h
        s.hCD,s.hCV,s.hHEX=head(),head(),head()
    def forward(s,x):
        g=s.ln(s.pool(s.enc(x)).flatten(1))
        return torch.cat([s.hCD(g),s.hCV(g),s.hHEX(g)],1)
base=models.CellposeModel(gpu=True, pretrained_model="cpsam_models/cpsam_v2")
net=Net(base.net.encoder.float()).to(dev)
import torch.utils.checkpoint as cp
for blk in net.enc.blocks:
    blk._of=blk.forward
    blk.forward=(lambda m: (lambda x: cp.checkpoint(m._of, x, use_reentrant=False)))(blk)
opt=torch.optim.AdamW([{"params":net.enc.parameters(),"lr":1e-5},
    {"params":list(net.hCD.parameters())+list(net.hCV.parameters())+list(net.hHEX.parameters()),"lr":1e-3}],weight_decay=1e-4)
Zt=torch.tensor(Z,device=dev)
sched=torch.optim.lr_scheduler.LambdaLR(opt, lambda step: min(1.0,(step+1)/200))
def evalm(idx):
    net.eval(); es=[]
    with torch.no_grad():
        for i in idx:
            x=tbatch([i],False)
            with torch.autocast("cuda",dtype=torch.bfloat16): p=net(x).float().mean(0).cpu().numpy()
            y=np.exp(p*LS+LM); g=Y[i]; es.append(np.abs(y-g)/g*100)
    return np.array(es).mean(0)
best=1e9; bp=f"{ROOT}/regress_ft_best_{TAG}.pt"; nb=len(tr)//BS
VA_FAST=va[:250]
for ep in range(EPOCHS):
    net.train(); np.random.shuffle(tr); t0=time.time(); tl=0
    for k in range(nb):
        b=tr[k*BS:(k+1)*BS]
        with torch.autocast("cuda",dtype=torch.bfloat16): p=net(tbatch(b,True))
        loss=(p-Zt[b]).abs().mean(); opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step(); sched.step(); tl+=loss.item()
    vm=evalm(VA_FAST); vov=vm.mean()
    print(f"ep{ep} trloss{tl/nb:.3f} val CD{vm[0]:.1f} CV{vm[1]:.1f} HEX{vm[2]:.1f} OV{vov:.1f} {time.time()-t0:.0f}s",flush=True)
    if vov<best: best=vov; torch.save(net.state_dict(),bp)
net.load_state_dict(torch.load(bp))
vm=evalm(va); tm=evalm(te)
print(f"\n=== FINAL grid-eval ===\nVAL  CD{vm[0]:.2f} CV{vm[1]:.2f} HEX{vm[2]:.2f} OV{vm.mean():.2f}\nTEST CD{tm[0]:.2f} CV{tm[1]:.2f} HEX{tm[2]:.2f} OV{tm.mean():.2f}",flush=True)
json.dump({"tag":TAG,"val":vm.tolist(),"test":tm.tolist(),"best_val_fast":float(best)},
          open(f"{ROOT}/regress_ft_results_{TAG}.json","w"),indent=1)
