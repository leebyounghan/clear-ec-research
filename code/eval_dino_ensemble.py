"""Does the DINOv2 member help the ensemble via decorrelation? Compute dino val/test preds with the
SAME C5x8 (40-view) TTA as the cached cpsam members, add to member_preds, redo per-metric greedy
selection, and report correlation of dino's errors with the current CV/HEX picks. Held-out split -> honest."""
import os, numpy as np, torch, torch.nn as nn, json, cv2
from transformers import AutoModel
ROOT="datasets/clear_ec_official"; SZ=512; CROP=256; DINO=224; dev="cuda"
LM=np.array([7.919919967651367,-0.9824771285057068,-0.6211431622505188],np.float32)
LS=np.array([0.18971863389015198,0.18813571333885193,0.2748945653438568],np.float32)
IMEAN=np.array([0.485,0.456,0.406],np.float32); ISTD=np.array([0.229,0.224,0.225],np.float32)
_clahe=cv2.createCLAHE(2.0,(8,8))
meta=np.load(f"{ROOT}/cache_native_meta_{SZ}.npz",allow_pickle=True); Y=meta["Y"].astype(np.float32); N=int(meta["N"])
imgs=np.array(np.memmap(f"{ROOT}/img_native_{SZ}.dat",dtype=np.uint8,mode='r',shape=(N,SZ,SZ)))
z=np.load(f"{ROOT}/member_preds.npz",allow_pickle=True)
va=z["va_idx"]; te=z["te_idx"]; Gva=z["gt_va"].astype(float); Gte=z["gt_te"].astype(float)
C5=[(0,0),(0,256),(256,0),(256,256),(128,128)]
def dih(a,k): return np.rot90(a[:,::-1] if k>=4 else a, k%4)
def prep(u):
    u=_clahe.apply(np.ascontiguousarray(u)); lo,hi=np.percentile(u,(1,99))
    u=np.clip((u.astype(np.float32)-lo)/(hi-lo+1e-6),0,1)
    u=cv2.resize(u,(DINO,DINO),interpolation=cv2.INTER_AREA)
    return ((np.stack([u]*3,0)-IMEAN[:,None,None])/ISTD[:,None,None]).astype(np.float32)
class DinoReg(nn.Module):
    def __init__(s):
        super().__init__(); s.bb=AutoModel.from_pretrained("facebook/dinov2-large")
        D=s.bb.config.hidden_size; s.ln=nn.LayerNorm(D)
        def h(): return nn.Sequential(nn.Linear(D,256),nn.GELU(),nn.Dropout(.2),nn.Linear(256,1))
        s.hCD,s.hCV,s.hHEX=h(),h(),h()
    def forward(s,x): g=s.ln(s.bb(pixel_values=x).pooler_output); return torch.cat([s.hCD(g),s.hCV(g),s.hHEX(g)],1)
net=DinoReg().to(dev); net.load_state_dict(torch.load(f"{ROOT}/regress_ft_best_dino.pt")); net.eval()
@torch.no_grad()
def preds(idx):
    out=np.zeros((len(idx),3),np.float32)
    for n,i in enumerate(idx):
        xs=[]
        for r,c in C5:
            w=_clahe.apply(np.ascontiguousarray(imgs[i][r:r+CROP,c:c+CROP])); lo,hi=np.percentile(w,(1,99))
            w=np.clip((w.astype(np.float32)-lo)/(hi-lo+1e-6),0,1); w=cv2.resize(w,(DINO,DINO),interpolation=cv2.INTER_AREA)
            for k in range(8):
                d=np.ascontiguousarray(dih(w,k)); xs.append(((np.stack([d]*3,0)-IMEAN[:,None,None])/ISTD[:,None,None]).astype(np.float32))
        x=torch.from_numpy(np.stack(xs)).to(dev)
        with torch.autocast("cuda",dtype=torch.bfloat16): p=net(x).float().mean(0).cpu().numpy()
        out[n]=np.exp(p*LS+LM)
    return out
dv=preds(va); dt=preds(te)
def mape(p,g): return float(np.mean(np.abs(p-g)/g*100))
names=["s2","at","cl","xHEX","ff","dino"]
Pv={n:(dv if n=="dino" else z[f"{n}_va"].astype(float)) for n in names}
Pt={n:(dt if n=="dino" else z[f"{n}_te"].astype(float)) for n in names}
ok_va=(Gva>0).all(1); ok_te=(Gte>0).all(1); M=["CD","CV","HEX"]
print("[dino] val",[round(mape(dv[ok_va,m],Gva[ok_va,m]),2) for m in range(3)],"test",[round(mape(dt[ok_te,m],Gte[ok_te,m]),2) for m in range(3)])
# error correlation of dino vs s2 (the CV pick) and at (HEX pick)
for m,mn in enumerate(M):
    es2=(Pv["s2"][ok_va,m]-Gva[ok_va,m]); ed=(dv[ok_va,m]-Gva[ok_va,m])
    print(f"  {mn}: corr(dino_err, s2_err)={np.corrcoef(es2,ed)[0,1]:.3f}")
print("\n=== greedy per-metric WITH dino ===")
val_f=[];test_f=[];pick={}
for m in range(3):
    sel=[];cur=1e9
    while True:
        best=None
        for n in names:
            if n in sel: continue
            e=mape(np.mean([Pv[j][ok_va,m] for j in sel+[n]],0),Gva[ok_va,m])
            if e<cur-1e-4: cur=e;best=n
        if best is None: break
        sel.append(best)
    pick[m]=sel
    pv=np.mean([Pv[j][ok_va,m] for j in sel],0); pt=np.mean([Pt[j][ok_te,m] for j in sel],0)
    val_f.append(mape(pv,Gva[ok_va,m])); test_f.append(mape(pt,Gte[ok_te,m]))
    print(f"{M[m]}: {'+'.join(sel)}  VAL {val_f[-1]:.2f} TEST {test_f[-1]:.2f}")
print(f"OV: VAL {np.mean(val_f):.2f} TEST {np.mean(test_f):.2f}  (without dino was VAL 8.99 TEST 8.92)")
np.savez_compressed(f"{ROOT}/dino_preds.npz",dino_va=dv,dino_te=dt,va_idx=va,te_idx=te)
json.dump({"pick":{M[m]:pick[m] for m in range(3)},"val":val_f,"test":test_f},open(f"{ROOT}/ens_dino_results.json","w"),indent=1)
