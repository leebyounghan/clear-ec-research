"""Cache per-member val+test predictions to npz (so ensemble selection is rerunnable without GPU).
Members: the 4 in the v3 container (s2/at/cl/xHEX, central-512 C5x8 TTA) + fullframe (24-grid).
Saves member_preds.npz: {name}_va (Nva,3), {name}_te (Nte,3) in ORIGINAL metric units + va_idx/te_idx."""
import os, json, numpy as np, torch, torch.nn as nn, cv2
from cellpose import models
ROOT="datasets/clear_ec_official"; SZ=512; CROP=256; H,W=972,1296; dev="cuda"
LM=np.array([7.919919967651367,-0.9824771285057068,-0.6211431622505188],np.float32)
LS=np.array([0.18971863389015198,0.18813571333885193,0.2748945653438568],np.float32)
meta=np.load(f"{ROOT}/cache_native_meta_{SZ}.npz",allow_pickle=True); Y=meta["Y"].astype(np.float32); N=int(meta["N"])
native=np.array(np.memmap(f"{ROOT}/img_native_{SZ}.dat",dtype=np.uint8,mode='r',shape=(N,SZ,SZ)))
rng=np.random.default_rng(42); perm=rng.permutation(N); te,va,tr=perm[:1000],perm[1000:2000],perm[2000:]
ymean=Y[tr].mean(0)
_cl=cv2.createCLAHE(2.0,(8,8))
def norm99(u):
    u=_cl.apply(np.clip(u,0,255).astype(np.uint8)).astype(np.float32)
    lo,hi=np.percentile(u,(1,99)); return np.clip((u-lo)/(hi-lo+1e-6),0,1)
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
        else: g=s.pool(f).flatten(1)
        g=s.ln(g); return torch.cat([s.hCD(g),s.hCV(g),s.hHEX(g)],1)
class SpecNet(nn.Module):
    def __init__(s,enc):
        super().__init__(); s.enc=enc; s.pool=nn.AdaptiveAvgPool2d(1); s.ln=nn.LayerNorm(256)
        def h(): return nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Dropout(.2),nn.Linear(128,1))
        s.hCD,s.hCV,s.hHEX=h(),h(),h()
    def forward(s,x): g=s.ln(s.pool(s.enc(x)).flatten(1)); return torch.cat([s.hCD(g),s.hCV(g),s.hHEX(g)],1)+1.0
base=models.CellposeModel(gpu=True, pretrained_model="cpsam_models/cpsam_v2")
fullimgs=None; GRID=[(r,c) for r in np.linspace(0,H-CROP,4).astype(int) for c in np.linspace(0,W-CROP,6).astype(int)]
def views_c5(i):
    out=[]
    for r,c in C5:
        w=norm99(native[i][r:r+CROP,c:c+CROP])
        for k in range(8): out.append(np.stack([np.ascontiguousarray(dih(w,k))]*3,0))
    return np.stack(out)
def views_grid(i):
    return np.stack([np.stack([norm99(np.array(fullimgs[i][r:r+CROP,c:c+CROP]))]*3,0) for r,c in GRID])
@torch.no_grad()
def run(net,idx,space,vf):
    out=np.zeros((len(idx),3),np.float32)
    for s in range(0,len(idx),2):
        b=idx[s:s+2]; x=np.concatenate([vf(i) for i in b],0)
        with torch.autocast("cuda",dtype=torch.bfloat16): p=net(torch.from_numpy(x).to(dev)).float()
        p=p.view(len(b),-1,3).mean(1)
        out[s:s+len(b)]=(np.exp(p.cpu().numpy()*LS+LM) if space=="log" else p.cpu().numpy()*ymean)
    return out
MEMBERS={ "s2":("regress_ft_best_clahe_s2.pt","mean","log","c5"),
          "at":("regress_ft_best_attn_clahe.pt","attn","log","c5"),
          "cl":("regress_ft_best_clahe_long.pt","mean","log","c5"),
          "xHEX":("regress_ft_best_spec_hex.pt","mean","spec","c5"),
          "ff":("regress_ft_best_fullframe.pt","mean","log","grid") }
out={"va_idx":va,"te_idx":te,"gt_va":Y[va],"gt_te":Y[te]}
for name,(ck,pool,space,vw) in MEMBERS.items():
    if vw=="grid" and fullimgs is None:
        fullimgs=np.memmap(f"{ROOT}/img_full_raw.dat",dtype=np.uint8,mode='r',shape=(N,H,W))
    net=(SpecNet(base.net.encoder.float()) if space=="spec" else ScalarNet(base.net.encoder.float(),pool)).to(dev)
    net.load_state_dict(torch.load(f"{ROOT}/{ck}"),strict=False); net.eval()
    vf=views_grid if vw=="grid" else views_c5; sp="log" if space=="log" else "ym"
    out[f"{name}_va"]=run(net,va,sp if sp=="log" else "ym",vf)
    out[f"{name}_te"]=run(net,te,sp if sp=="log" else "ym",vf)
    e=np.abs(out[f"{name}_va"]-Y[va])/np.clip(Y[va],1e-9,None)*100
    print(f"[{name}] val CD{e[:,0].mean():.2f} CV{e[:,1].mean():.2f} HEX{e[:,2].mean():.2f}",flush=True)
    del net; torch.cuda.empty_cache()
np.savez_compressed(f"{ROOT}/member_preds.npz",**out)
print("saved -> member_preds.npz",flush=True)
