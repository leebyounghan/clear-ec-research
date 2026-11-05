"""Per-metric ensemble: the 6 diverse multitask models (ens_final) + 3 single-task SPECIALISTS
(spec_cd/cv/hex). For each metric INDEPENDENTLY, greedily select the subset of members whose averaged
prediction minimises VAL MAPE for that metric (specialists compete per-metric). Report TEST. Select on
val only, no test tuning. Beats the joint-OV ens_final if specialists add per-metric signal."""
import os, json, numpy as np, torch, torch.nn as nn, cv2
from cellpose import models
ROOT="datasets/clear_ec_official"; SZ=512; CROP=256; dev="cuda"
meta=np.load(f"{ROOT}/cache_native_meta_{SZ}.npz",allow_pickle=True); Y=meta["Y"].astype(np.float32); N=int(meta["N"])
native=np.array(np.memmap(f"{ROOT}/img_native_{SZ}.dat",dtype=np.uint8,mode='r',shape=(N,SZ,SZ)))
clahe_full=np.array(np.memmap(f"{ROOT}/img_clahe_{SZ}.dat",dtype=np.uint8,mode='r',shape=(N,SZ,SZ)))
rng=np.random.default_rng(42); perm=rng.permutation(N); te,va,tr=perm[:1000],perm[1000:2000],perm[2000:]
logY=np.log(np.clip(Y,1e-6,None)); lm=logY[tr].mean(0); ls=logY[tr].std(0)
ls_t=torch.tensor(ls,device=dev); lm_t=torch.tensor(lm,device=dev); ymean=Y[tr].mean(0)
_cl=cv2.createCLAHE(2.0,(8,8))
def norm99_raw(u,clahe):
    if clahe: u=_cl.apply(np.clip(u,0,255).astype(np.uint8)).astype(np.float32)
    lo,hi=np.percentile(u,(1,99)); return np.clip((u.astype(np.float32)-lo)/(hi-lo+1e-6),0,1)
def norm99_full(u):
    lo,hi=np.percentile(u,(1,99)); return np.clip((u.astype(np.float32)-lo)/(hi-lo+1e-6),0,1)
C5=[(0,0),(0,256),(256,0),(256,256),(128,128)]
def dih(a,k): return np.rot90(a[:,::-1] if k>=4 else a, k%4)
class ScalarNet(nn.Module):
    def __init__(s,enc,pool):
        super().__init__(); s.enc=enc; s.pool=nn.AdaptiveAvgPool2d(1); D=512 if pool=="meanstd" else 256; s.ln=nn.LayerNorm(D); s.P=pool
        if pool=="attn": s.qtok=nn.Parameter(torch.zeros(1,1,256)); s.attn=nn.MultiheadAttention(256,4,batch_first=True)
        def h(): return nn.Sequential(nn.Linear(D,128),nn.GELU(),nn.Dropout(.2),nn.Linear(128,1))
        s.hCD,s.hCV,s.hHEX=h(),h(),h()
        def up(i,o): return nn.Sequential(nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False),nn.Conv2d(i,o,3,1,1),nn.GroupNorm(8,o),nn.GELU())
        s.dec=nn.Sequential(up(256,128),up(128,64),up(64,32),nn.Conv2d(32,1,1))
    def forward(s,x):
        f=s.enc(x); B=f.shape[0]
        if s.P=="attn": seq=f.flatten(2).transpose(1,2); g=s.attn(s.qtok.expand(B,-1,-1),seq,seq)[0].squeeze(1)
        elif s.P=="meanstd": v=f.flatten(2); g=torch.cat([v.mean(-1),v.std(-1)],1)
        else: g=s.pool(f).flatten(1)
        g=s.ln(g); return torch.cat([s.hCD(g),s.hCV(g),s.hHEX(g)],1)
class SpecNet(nn.Module):  # = train_finetune Net (avgpool, +1.0, ymean-space)
    def __init__(s,enc):
        super().__init__(); s.enc=enc; s.pool=nn.AdaptiveAvgPool2d(1); s.ln=nn.LayerNorm(256)
        def h(): return nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Dropout(.2),nn.Linear(128,1))
        s.hCD,s.hCV,s.hHEX=h(),h(),h()
    def forward(s,x): g=s.ln(s.pool(s.enc(x)).flatten(1)); return torch.cat([s.hCD(g),s.hCV(g),s.hHEX(g)],1)+1.0
base=models.CellposeModel(gpu=True, pretrained_model="cpsam_models/cpsam_v2")
def crops(i,src,clahe):
    out=[]; img=clahe_full[i] if src=="full" else native[i]
    for r,c in C5:
        b=norm99_full(img[r:r+CROP,c:c+CROP]) if src=="full" else norm99_raw(img[r:r+CROP,c:c+CROP],clahe)
        for k in range(8): out.append(np.stack([dih(b,k).copy()]*3,0))
    return np.stack(out)
@torch.no_grad()
def preds(net,src,clahe,idx,logspace):
    out=np.zeros((len(idx),3),np.float32)
    for s in range(0,len(idx),2):
        b=idx[s:s+2]; x=np.concatenate([crops(i,src,clahe) for i in b],0)
        with torch.autocast("cuda",dtype=torch.bfloat16): p=net(torch.from_numpy(x).to(dev)).float()
        p=p.view(len(b),40,3).mean(1)
        out[s:s+len(b)]=(torch.exp(p*ls_t+lm_t).cpu().numpy() if logspace else p.cpu().numpy()*ymean)
    return out
MT={ "s2":("regress_ft_best_clahe_s2.pt","raw",1,"mean"),"cl":("regress_ft_best_clahe_long.pt","raw",1,"mean"),
     "rl":("regress_ft_best_robust_long.pt","raw",0,"mean"),"at":("regress_ft_best_attn_clahe.pt","raw",1,"attn"),
     "mt":("regress_ft_best_mtl_l05.pt","full",0,"mean"),"sp":("regress_ft_best_stdpool_ms.pt","full",0,"meanstd") }
SPEC={ "xCD":("regress_ft_best_spec_cd.pt",0),"xCV":("regress_ft_best_spec_cv.pt",1),"xHEX":("regress_ft_best_spec_hex.pt",2) }
Pv={}; Pt={}
for tag,(ck,src,cl,pool) in MT.items():
    net=ScalarNet(base.net.encoder.float(),pool).to(dev); net.load_state_dict(torch.load(f"{ROOT}/{ck}"),strict=False); net.eval()
    Pv[tag]=preds(net,src,cl,va,True); Pt[tag]=preds(net,src,cl,te,True); del net; torch.cuda.empty_cache()
    e=np.abs(Pv[tag]-Y[va])/Y[va]*100; print(f"[{tag}] val CD{e[:,0].mean():.1f} CV{e[:,1].mean():.1f} HEX{e[:,2].mean():.1f}",flush=True)
for tag,(ck,mi) in SPEC.items():
    if not os.path.exists(f"{ROOT}/{ck}"): print(f"[{tag}] missing, skip"); continue
    net=SpecNet(base.net.encoder.float()).to(dev); net.load_state_dict(torch.load(f"{ROOT}/{ck}"),strict=False); net.eval()
    Pv[tag]=preds(net,"raw",1,va,False); Pt[tag]=preds(net,"raw",1,te,False); del net; torch.cuda.empty_cache()
    e=np.abs(Pv[tag][:,mi]-Y[va][:,mi])/Y[va][:,mi]*100; print(f"[{tag}] val metric{mi} {e.mean():.2f}",flush=True)
def me(P,idx,m): return float(np.mean(np.abs(P-Y[idx][:,m])/Y[idx][:,m]*100))
names=["CD","CV","HEX"]; test_final=[]; val_final=[]
for m in range(3):
    cand=[k for k in Pv]  # all members have a col m; specialists reliable only on their metric but let selection decide
    sel=[]; cur=None; curerr=1e9
    while True:
        bestadd=None
        for k in cand:
            if k in sel: continue
            arr=np.mean([Pv[j][:,m] for j in sel+[k]],0); e=me(arr,va,m)
            if e<curerr-1e-4: curerr=e; bestadd=k
        if bestadd is None: break
        sel.append(bestadd); cur=curerr
    vt=me(np.mean([Pv[j][:,m] for j in sel],0),va,m); tt=me(np.mean([Pt[j][:,m] for j in sel],0),te,m)
    val_final.append(vt); test_final.append(tt)
    print(f"metric {names[m]}: members={'+'.join(sel)}  VAL {vt:.2f}  TEST {tt:.2f}",flush=True)
print(f"\n=== PER-METRIC ENSEMBLE ===\nVAL  CD{val_final[0]:.2f} CV{val_final[1]:.2f} HEX{val_final[2]:.2f} OV{np.mean(val_final):.2f}")
print(f"TEST CD{test_final[0]:.2f} CV{test_final[1]:.2f} HEX{test_final[2]:.2f} OV{np.mean(test_final):.2f}  (ens_final was 8.95)",flush=True)
json.dump({"val":val_final,"test":test_final,"ov_test":float(np.mean(test_final))},open(f"{ROOT}/ens_specialist_results.json","w"),indent=1)
