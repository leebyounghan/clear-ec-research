"""Fine-tune Cellpose-SAM ViT encoder + 3 per-metric heads (CD/CV/HEX).
Input: 256x256 NATIVE-res crops (cells ~27px = cpsam scale) from central-512 cache.
Split: 1000 test / 1000 val / rest train (naive random seed=42)."""
import os, numpy as np, torch, torch.nn as nn, json, time, cv2
from cellpose import models
ROOT="datasets/clear_ec_official"; SZ=512; CROP=256
STEPS=int(os.environ.get("PROBE_STEPS","0")); EPOCHS=int(os.environ.get("EPOCHS","25")); BS=int(os.environ.get("BS","16"))
CLAHE=int(os.environ.get("CLAHE","0")); TAG=os.environ.get("TAG","base")
_clahe=cv2.createCLAHE(2.0,(8,8))
dev="cuda"
meta=np.load(f"{ROOT}/cache_native_meta_{SZ}.npz", allow_pickle=True)
Y=meta["Y"].astype(np.float32); N=int(meta["N"])
imgs=np.array(np.memmap(f"{ROOT}/img_native_{SZ}.dat",dtype=np.uint8,mode='r',shape=(N,SZ,SZ)))
rng=np.random.default_rng(42); perm=rng.permutation(N)
te,va,tr=perm[:1000],perm[1000:2000],perm[2000:]
ymean=Y[tr].mean(0); Yn=Y/ymean
print(f"split train {len(tr)} val {len(va)} test {len(te)} ymean {ymean}",flush=True)

def norm99(u):
    if CLAHE: u=_clahe.apply(u.astype(np.uint8))
    lo,hi=np.percentile(u,(1,99)); return np.clip((u.astype(np.float32)-lo)/(hi-lo+1e-6),0,1)
CROPS5=[(0,0),(0,256),(256,0),(256,256),(128,128)]
def make(i, train):
    if train:
        r,c=np.random.randint(0,SZ-CROP+1),np.random.randint(0,SZ-CROP+1)
        im=norm99(imgs[i][r:r+CROP,c:c+CROP])
        if np.random.rand()<.5: im=im[:,::-1]
        if np.random.rand()<.5: im=im[::-1,:]
        im=np.rot90(im,np.random.randint(4))
        return np.stack([im]*3,0)[None]
    else:
        outs=[np.stack([norm99(imgs[i][r:r+CROP,c:c+CROP])]*3,0) for r,c in CROPS5]
        return np.stack(outs)   # (5,3,256,256)
def tbatch(idx, train): 
    return torch.from_numpy(np.concatenate([make(i,train) for i in idx],0)).to(dev)

class Net(nn.Module):
    def __init__(s, enc):
        super().__init__(); s.enc=enc; s.pool=nn.AdaptiveAvgPool2d(1); s.ln=nn.LayerNorm(256)
        def head():
            h=nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Dropout(.2),nn.Linear(128,1))
            nn.init.zeros_(h[-1].weight); nn.init.zeros_(h[-1].bias)   # start pred=1.0 (=train mean)
            return h
        s.hCD,s.hCV,s.hHEX=head(),head(),head()
    def forward(s,x):
        g=s.ln(s.pool(s.enc(x)).flatten(1))
        return torch.cat([s.hCD(g),s.hCV(g),s.hHEX(g)],1)+1.0
base=models.CellposeModel(gpu=True, pretrained_model="cpsam_models/cpsam_v2")
net=Net(base.net.encoder.float()).to(dev)
# gradient checkpointing on each ViT block to cut activation memory
import torch.utils.checkpoint as cp
for blk in net.enc.blocks:
    blk._of=blk.forward
    blk.forward=(lambda m: (lambda x: cp.checkpoint(m._of, x, use_reentrant=False)))(blk)
opt=torch.optim.AdamW([{"params":net.enc.parameters(),"lr":1e-5},
    {"params":list(net.hCD.parameters())+list(net.hCV.parameters())+list(net.hHEX.parameters()),"lr":1e-3}],weight_decay=1e-4)
Ytr=torch.tensor(Yn,device=dev)
WARM=200
sched=torch.optim.lr_scheduler.LambdaLR(opt, lambda step: min(1.0,(step+1)/WARM))
def mloss(p,idx): y=Ytr[idx]; return ((p-y).abs()/y.clamp(min=1e-6)).mean()
def evalm(idx, five=False):
    net.eval(); es=[]
    with torch.no_grad():
        for s in range(0,len(idx),BS):
            b=idx[s:s+BS]
            if five:
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    p=net(tbatch(b,False)).float().view(len(b),5,3).mean(1)
            else:  # fast: single center crop
                xs=torch.from_numpy(np.stack([np.stack([norm99(imgs[i][128:384,128:384])]*3,0) for i in b])).to(dev)
                with torch.autocast("cuda",dtype=torch.bfloat16): p=net(xs).float()
            y=Ytr[b]; es.append(((p-y).abs()/y.clamp(min=1e-6)).cpu())
    return torch.cat(es).mean(0).numpy()*100

if STEPS>0:
    net.train()
    for k in range(STEPS):
        b=tr[k*BS:(k+1)*BS]
        with torch.autocast("cuda",dtype=torch.bfloat16): p=net(tbatch(b,True))
        loss=mloss(p,b); opt.zero_grad(); loss.backward(); opt.step()
    print(f"PROBE ok bs={BS} crop={CROP} peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB loss{loss.item():.3f}",flush=True); raise SystemExit

best=1e9; bp=f"{ROOT}/regress_ft_best_{TAG}.pt"; nb=len(tr)//BS
for ep in range(EPOCHS):
    net.train(); np.random.shuffle(tr); t0=time.time(); tl=0
    for k in range(nb):
        b=tr[k*BS:(k+1)*BS]
        with torch.autocast("cuda",dtype=torch.bfloat16): p=net(tbatch(b,True))
        loss=mloss(p,b); opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step(); sched.step(); tl+=loss.item()
    vm=evalm(va); vov=vm.mean()
    print(f"ep{ep} trloss{tl/nb:.3f} val CD{vm[0]:.1f} CV{vm[1]:.1f} HEX{vm[2]:.1f} OV{vov:.1f} {time.time()-t0:.0f}s",flush=True)
    if vov<best: best=vov; torch.save(net.state_dict(),bp)
net.load_state_dict(torch.load(bp))
tm=evalm(te,five=True); vm=evalm(va,five=True)
print(f"\n=== FINAL (best val OV {best:.1f}) ===\nVAL  CD{vm[0]:.1f} CV{vm[1]:.1f} HEX{vm[2]:.1f} OV{vm.mean():.1f}\nTEST CD{tm[0]:.1f} CV{tm[1]:.1f} HEX{tm[2]:.1f} OV{tm.mean():.1f}",flush=True)
json.dump({"tag":TAG,"clahe":CLAHE,"test":tm.tolist(),"val":vm.tolist(),"best_val":float(best)},open(f"{ROOT}/regress_ft_results_{TAG}.json","w"),indent=1)
