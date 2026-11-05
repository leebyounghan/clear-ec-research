import os, numpy as np, torch, torch.nn as nn, json, cv2
from cellpose import models
ROOT="datasets/clear_ec_official"; SZ=512; CROP=256; BS=16; dev="cuda"
TAG=os.environ["TAG"]; CLAHE=int(os.environ.get("CLAHE","0")); _cl=cv2.createCLAHE(2.0,(8,8))
meta=np.load(f"{ROOT}/cache_native_meta_{SZ}.npz",allow_pickle=True)
Y=meta["Y"].astype(np.float32); N=int(meta["N"])
imgs=np.array(np.memmap(f"{ROOT}/img_native_{SZ}.dat",dtype=np.uint8,mode='r',shape=(N,SZ,SZ)))
rng=np.random.default_rng(42); perm=rng.permutation(N); te,va,tr=perm[:1000],perm[1000:2000],perm[2000:]
ymean=Y[tr].mean(0); Yn=Y/ymean
def norm99(u):
    if CLAHE: u=_cl.apply(u.astype(np.uint8))
    lo,hi=np.percentile(u,(1,99)); return np.clip((u.astype(np.float32)-lo)/(hi-lo+1e-6),0,1)
C5=[(0,0),(0,256),(256,0),(256,256),(128,128)]
class Net(nn.Module):
    def __init__(s,enc):
        super().__init__(); s.enc=enc; s.pool=nn.AdaptiveAvgPool2d(1); s.ln=nn.LayerNorm(256)
        def h():
            m=nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Dropout(.2),nn.Linear(128,1)); return m
        s.hCD,s.hCV,s.hHEX=h(),h(),h()
    def forward(s,x):
        g=s.ln(s.pool(s.enc(x)).flatten(1)); return torch.cat([s.hCD(g),s.hCV(g),s.hHEX(g)],1)+1.0
base=models.CellposeModel(gpu=True, pretrained_model="cpsam_models/cpsam_v2")
net=Net(base.net.encoder.float()).to(dev)
net.load_state_dict(torch.load(f"{ROOT}/regress_ft_best_{TAG}.pt")); net.eval()
Ytr=torch.tensor(Yn,device=dev)
def evalm(idx):
    es=[]
    with torch.no_grad():
        for s in range(0,len(idx),BS):
            b=idx[s:s+BS]
            x=np.concatenate([np.stack([np.stack([norm99(imgs[i][r:r+CROP,c:c+CROP])]*3,0) for r,c in C5]) for i in b],0)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                p=net(torch.from_numpy(x).to(dev)).float().view(len(b),5,3).mean(1)
            y=Ytr[b]; es.append(((p-y).abs()/y.clamp(min=1e-6)).cpu())
    return torch.cat(es).mean(0).numpy()*100
tm=evalm(te); vm=evalm(va)
print(f"[{TAG}] TEST CD{tm[0]:.1f} CV{tm[1]:.1f} HEX{tm[2]:.1f} OV{tm.mean():.1f} | VAL OV{vm.mean():.1f}")
json.dump({"tag":TAG,"test":tm.tolist(),"val":vm.tolist()},open(f"{ROOT}/regress_ft_results_{TAG}.json","w"),indent=1)
