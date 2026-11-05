"""Decorrelated ensemble member: DINOv2-large backbone + per-metric regression heads.
Rationale: current 4 members all share the cpsam ViT encoder -> highly correlated. DINOv2 has a
completely different pretraining prior, so even at similar individual accuracy it decorrelates the
ensemble and shaves all 3 metrics (Bone-Age winner: decorrelated members > best-single naive combine).
Recipe kept parallel to clahe_s2: central-512 native cache, per-crop CLAHE, log-space targets, C5x8 TTA.
Input: 256 crop -> resize 224 (dino patch14, imagenet norm). Fine-tune backbone (low LR) + MLP heads.
YSRC=raw|smooth (uses Y_slidesmooth for train targets). Val-selected on held-out split (not all-9000);
the all-9000 final of the winning config comes later via train_final9000-style port."""
import os, numpy as np, torch, torch.nn as nn, json, time, cv2
from transformers import AutoModel
ROOT="datasets/clear_ec_official"; SZ=512; CROP=256; DINO=224
EPOCHS=int(os.environ.get("EPOCHS","20")); BS=int(os.environ.get("BS","16"))
YSRC=os.environ.get("YSRC","smooth"); TAG=os.environ.get("TAG","dino"); dev="cuda"
LM=np.array([7.919919967651367,-0.9824771285057068,-0.6211431622505188],np.float32)
LS=np.array([0.18971863389015198,0.18813571333885193,0.2748945653438568],np.float32)
IMEAN=np.array([0.485,0.456,0.406],np.float32); ISTD=np.array([0.229,0.224,0.225],np.float32)
_clahe=cv2.createCLAHE(2.0,(8,8))
meta=np.load(f"{ROOT}/cache_native_meta_{SZ}.npz",allow_pickle=True); Y=meta["Y"].astype(np.float32); N=int(meta["N"])
Ytr_src=np.load(f"{ROOT}/Y_slidesmooth.npy").astype(np.float32) if YSRC=="smooth" else Y
imgs=np.array(np.memmap(f"{ROOT}/img_native_{SZ}.dat",dtype=np.uint8,mode='r',shape=(N,SZ,SZ)))
rng=np.random.default_rng(42); perm=rng.permutation(N); te,va,tr=perm[:1000],perm[1000:2000],perm[2000:]
bad=set(np.where((Ytr_src<=0).any(1))[0].tolist()); tr=np.array([i for i in tr if i not in bad])
Z=(np.log(np.clip(Ytr_src,1e-6,None))-LM)/LS
print(f"[{TAG}] DINOv2-large YSRC={YSRC} train {len(tr)} val {len(va)}",flush=True)

def prep(u):  # 256 crop -> CLAHE -> resize 224 -> imagenet-norm 3ch
    u=_clahe.apply(np.ascontiguousarray(u)); lo,hi=np.percentile(u,(1,99))
    u=np.clip((u.astype(np.float32)-lo)/(hi-lo+1e-6),0,1)
    u=cv2.resize(u,(DINO,DINO),interpolation=cv2.INTER_AREA)
    return ((np.stack([u]*3,0)-IMEAN[:,None,None])/ISTD[:,None,None]).astype(np.float32)
C5=[(0,0),(0,256),(256,0),(256,256),(128,128)]
def make(i,train):
    if train:
        r,c=np.random.randint(0,SZ-CROP+1),np.random.randint(0,SZ-CROP+1)
        u=imgs[i][r:r+CROP,c:c+CROP]
        if np.random.rand()<.5: u=u[:,::-1]
        if np.random.rand()<.5: u=u[::-1,:]
        u=np.ascontiguousarray(np.rot90(u,np.random.randint(4)))
        return prep(u)[None]
    return np.stack([prep(imgs[i][r:r+CROP,c:c+CROP]) for r,c in C5])
def tbatch(idx,train): return torch.from_numpy(np.concatenate([make(i,train) for i in idx],0)).to(dev)

class DinoReg(nn.Module):
    def __init__(s):
        super().__init__(); s.bb=AutoModel.from_pretrained("facebook/dinov2-large")
        s.bb.gradient_checkpointing_enable()
        D=s.bb.config.hidden_size; s.ln=nn.LayerNorm(D)
        def h():
            m=nn.Sequential(nn.Linear(D,256),nn.GELU(),nn.Dropout(.2),nn.Linear(256,1))
            nn.init.zeros_(m[-1].weight); nn.init.zeros_(m[-1].bias); return m
        s.hCD,s.hCV,s.hHEX=h(),h(),h()
    def forward(s,x):
        g=s.bb(pixel_values=x).pooler_output   # CLS-pooled
        g=s.ln(g); return torch.cat([s.hCD(g),s.hCV(g),s.hHEX(g)],1)
net=DinoReg().to(dev)
opt=torch.optim.AdamW([{"params":net.bb.parameters(),"lr":1e-5},
    {"params":list(net.hCD.parameters())+list(net.hCV.parameters())+list(net.hHEX.parameters())+list(net.ln.parameters()),"lr":1e-3}],weight_decay=1e-4)
Zt=torch.tensor(Z,device=dev)
sched=torch.optim.lr_scheduler.LambdaLR(opt, lambda st: min(1.0,(st+1)/200))
def evalm(idx):
    net.eval(); es=[]
    with torch.no_grad():
        for s in range(0,len(idx),4):
            b=idx[s:s+4]; x=tbatch(b,False)
            with torch.autocast("cuda",dtype=torch.bfloat16): p=net(x).float().view(len(b),5,3).mean(1).cpu().numpy()
            y=np.exp(p*LS+LM); g=Y[b]; es.append(np.abs(y-g)/np.clip(g,1e-9,None)*100)
    return np.concatenate(es).mean(0)
best=1e9; bp=f"{ROOT}/regress_ft_best_{TAG}.pt"; nb=len(tr)//BS
for ep in range(EPOCHS):
    net.train(); np.random.shuffle(tr); t0=time.time(); tl=0
    for k in range(nb):
        b=tr[k*BS:(k+1)*BS]
        with torch.autocast("cuda",dtype=torch.bfloat16): p=net(tbatch(b,True))
        loss=(p-Zt[b]).abs().mean(); opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step(); sched.step(); tl+=loss.item()
    vm=evalm(va[:250]); vov=vm.mean()
    print(f"ep{ep} trloss{tl/nb:.3f} val CD{vm[0]:.1f} CV{vm[1]:.1f} HEX{vm[2]:.1f} OV{vov:.1f} {time.time()-t0:.0f}s",flush=True)
    if vov<best: best=vov; torch.save(net.state_dict(),bp)
net.load_state_dict(torch.load(bp)); vm=evalm(va); tm=evalm(te)
print(f"\n=== FINAL [{TAG}] ===\nVAL CD{vm[0]:.2f} CV{vm[1]:.2f} HEX{vm[2]:.2f} OV{vm.mean():.2f}\nTEST CD{tm[0]:.2f} CV{tm[1]:.2f} HEX{tm[2]:.2f} OV{tm.mean():.2f}",flush=True)
json.dump({"tag":TAG,"val":vm.tolist(),"test":tm.tolist()},open(f"{ROOT}/regress_ft_results_{TAG}.json","w"),indent=1)
