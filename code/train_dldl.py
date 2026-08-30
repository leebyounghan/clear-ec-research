"""DLDL-style distribution regression for CD/CV/HEX (research top-pick for weak CV/HEX).
Each metric -> B bins in LOG space; target = Gaussian label distribution around true log-value;
predict softmax over bins; loss = KL + lambda*|E[pred]-true| (expectation matched). Predicted
value = exp(E_bin). Robust + captures uncertainty. Model select on VAL; TEST held-out."""
import os, json, numpy as np, torch, torch.nn as nn, time, math
from cellpose import models
ROOT="datasets/clear_ec_official"; SZ=512; CROP=256; dev="cuda"
EPOCHS=int(os.environ.get("EPOCHS","12")); BS=int(os.environ.get("BS","16")); B=int(os.environ.get("BINS","50")); TAG=os.environ.get("TAG","dldl")
meta=np.load(f"{ROOT}/cache_native_meta_{SZ}.npz",allow_pickle=True)
Y=meta["Y"].astype(np.float32); N=int(meta["N"])
imgs=np.array(np.memmap(f"{ROOT}/img_native_{SZ}.dat",dtype=np.uint8,mode='r',shape=(N,SZ,SZ)))
rng=np.random.default_rng(42); perm=rng.permutation(N); te,va,tr=perm[:1000],perm[1000:2000],perm[2000:]
logY=np.log(np.clip(Y,1e-6,None))
lo=np.percentile(logY[tr],0.5,axis=0); hi=np.percentile(logY[tr],99.5,axis=0)   # robust range per metric
centers=np.stack([np.linspace(lo[j],hi[j],B) for j in range(3)],1)               # (B,3)
sig=(hi-lo)/B*2.0                                                                  # label-dist width
cent_t=torch.tensor(centers,device=dev,dtype=torch.float32)                        # (B,3)
# soft target distribution per sample per metric: (N,3,B)
def soft_target(logy):  # logy (m,3)
    d=logy[:,None,:]-centers[None,:,:]                                             # (m,B,3)
    w=np.exp(-(d**2)/(2*sig[None,None,:]**2)); w/=w.sum(1,keepdims=True)+1e-9
    return w.astype(np.float32)                                                    # (m,B,3)
ST=soft_target(logY); ST_t=torch.tensor(ST,device=dev)
Yreal_t=torch.tensor(Y,device=dev)
def norm99(u):
    lo2,hi2=np.percentile(u,(1,99)); return np.clip((u.astype(np.float32)-lo2)/(hi2-lo2+1e-6),0,1)
def crop(i,train):
    if train:
        r,c=np.random.randint(0,SZ-CROP+1),np.random.randint(0,SZ-CROP+1); im=norm99(imgs[i][r:r+CROP,c:c+CROP])
        if np.random.rand()<.5: im=im[:,::-1]
        if np.random.rand()<.5: im=im[::-1]
        im=np.rot90(im,np.random.randint(4)); return np.stack([im]*3,0)[None]
    C5=[(0,0),(0,256),(256,0),(256,256),(128,128)]
    return np.stack([np.stack([norm99(imgs[i][r:r+CROP,c:c+CROP])]*3,0) for r,c in C5])
class Net(nn.Module):
    def __init__(s, enc):
        super().__init__(); s.enc=enc; s.pool=nn.AdaptiveAvgPool2d(1); s.ln=nn.LayerNorm(256)
        def h(): return nn.Sequential(nn.Linear(256,256),nn.GELU(),nn.Dropout(.2),nn.Linear(256,B))
        s.hCD,s.hCV,s.hHEX=h(),h(),h()
    def forward(s,x):
        f=s.enc(x); g=s.ln(s.pool(f).flatten(1))
        return torch.stack([s.hCD(g),s.hCV(g),s.hHEX(g)],2)   # (Bt,B,3) logits over bins
base=models.CellposeModel(gpu=True, pretrained_model="cpsam_models/cpsam_v2")
net=Net(base.net.encoder.float()).to(dev)
import torch.utils.checkpoint as cp
for blk in net.enc.blocks:
    blk._of=blk.forward; blk.forward=(lambda m:(lambda x:cp.checkpoint(m._of,x,use_reentrant=False)))(blk)
opt=torch.optim.AdamW([{"params":net.enc.parameters(),"lr":1e-5},
    {"params":list(net.hCD.parameters())+list(net.hCV.parameters())+list(net.hHEX.parameters()),"lr":1e-3}],weight_decay=1e-4)
WARM=200; nb=len(tr)//BS; TOTAL=EPOCHS*nb
def lr_lambda(s):
    if s<WARM: return (s+1)/WARM
    p=(s-WARM)/max(1,TOTAL-WARM); return 0.5*(1+math.cos(math.pi*min(1.0,p)))
sched=torch.optim.lr_scheduler.LambdaLR(opt,lr_lambda)
def expect_real(logits):  # (m,B,3) -> real value via softmax expectation in log space
    p=torch.softmax(logits,1); e=(p*cent_t[None]).sum(1); return torch.exp(e)   # (m,3)
def dldl_loss(logits, idx):
    logp=torch.log_softmax(logits,1); tgt=ST_t[idx].permute(0,2,1).reshape(logits.shape[0],B,3) if False else ST_t[idx]  # ST_t idx (m,B,3)
    kl=-(tgt*logp).sum(1).mean()                                   # cross-entropy to soft target
    e=(torch.softmax(logits,1)*cent_t[None]).sum(1); l1=(e-torch.tensor(logY[idx],device=dev)).abs().mean()
    return kl+l1
def evalm(idx):
    net.eval(); out=np.zeros((len(idx),3),np.float32)
    with torch.no_grad():
        for s in range(0,len(idx),BS):
            b=idx[s:s+BS]
            x=np.concatenate([crop(i,False) for i in b],0)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                lg=net(torch.from_numpy(x).to(dev)).float().view(len(b),5,B,3).mean(1)
            out[s:s+len(b)]=expect_real(lg).cpu().numpy()
    return np.mean(np.abs(out-Y[idx])/Y[idx]*100,0)
best=1e9; bp=f"{ROOT}/regress_ft_best_{TAG}.pt"
for ep in range(EPOCHS):
    net.train(); np.random.shuffle(tr); t0=time.time(); tl=0
    for k in range(nb):
        b=tr[k*BS:(k+1)*BS]; x=torch.from_numpy(np.concatenate([crop(i,True) for i in b],0)).to(dev)
        with torch.autocast("cuda",dtype=torch.bfloat16): lg=net(x); loss=dldl_loss(lg,b)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step(); sched.step(); tl+=loss.item()
    vm=evalm(va); vov=vm.mean()
    print(f"ep{ep} loss{tl/nb:.3f} | val CD{vm[0]:.1f} CV{vm[1]:.1f} HEX{vm[2]:.1f} OV{vov:.2f} {time.time()-t0:.0f}s",flush=True)
    if vov<best: best=vov; torch.save(net.state_dict(),bp)
net.load_state_dict(torch.load(bp)); tm=evalm(te); vm=evalm(va)
print(f"\n=== {TAG} FINAL (best val {best:.2f}) ===\nVAL  CD{vm[0]:.1f} CV{vm[1]:.1f} HEX{vm[2]:.1f} OV{vm.mean():.2f}\nTEST CD{tm[0]:.1f} CV{tm[1]:.1f} HEX{tm[2]:.1f} OV{tm.mean():.2f}",flush=True)
json.dump({"test":tm.tolist(),"val":vm.tolist(),"best_val":float(best)},open(f"{ROOT}/regress_ft_results_{TAG}.json","w"),indent=1)
