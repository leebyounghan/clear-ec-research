"""SSL round-1, step 2+3: train a small U-Net to predict fNLA-style cell edges from raw, using
pool pseudo-labels. TWO ARMS via env TAG: 'ungated' (all pool) vs 'gated' (scalar-MPE<=THR).
Target = fNLA edge_prob (soft). Train on in-focus-box 256 crops (periphery pseudo-edges = noise).
Then eval on the 25 green: U-Net edge -> watershed+trim -> downstream CD/CV/HEX MPE.
"""
import os, json, csv, numpy as np, torch, torch.nn as nn, cv2, SimpleITK as sitk
from skimage.measure import regionprops
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.draw import polygon
from scipy import ndimage as ndi
ROOT="datasets/clear_ec_official"; OUT=f"{ROOT}/green_eval"; dev="cuda"
TAG=os.environ.get("TAG","gated"); THR=float(os.environ.get("THR","12")); STEPS=int(os.environ.get("STEPS","2500"))
IW,IH=1296,972; PH=976
m=np.load(f"{ROOT}/pool_meta.npz",allow_pickle=True); ids=list(m["ids"]); N=int(m["N"]); mpe=m["mpe"]; BOX=m["box"]
EU=np.memmap(f"{ROOT}/pool_edge_u8.dat",dtype=np.uint8,mode='r',shape=(N,IH,IW))
worst=mpe.max(1)
sel=np.arange(N) if TAG=="ungated" else np.where(worst<=THR)[0]
print(f"TAG={TAG} THR={THR} -> {len(sel)} training images",flush=True)
# load raw for selected into RAM (uint8)
MDIR=f"{ROOT}/train_mha"
def readmha(p):
    a=sitk.GetArrayFromImage(sitk.ReadImage(p)).astype(np.float32); return a[0] if a.ndim==3 else a
RAW={}
for i in sel: RAW[i]=readmha(f"{MDIR}/{ids[i]}.mha")
print("raw loaded",flush=True)
def std(u): return (u-u.mean())/(u.std()+1e-6)

# ---- U-Net ----
def cbr(i,o): return nn.Sequential(nn.Conv2d(i,o,3,1,1),nn.BatchNorm2d(o),nn.ReLU(True),
                                   nn.Conv2d(o,o,3,1,1),nn.BatchNorm2d(o),nn.ReLU(True))
class UNet(nn.Module):
    def __init__(s,b=32):
        super().__init__(); s.d1=cbr(1,b); s.d2=cbr(b,b*2); s.d3=cbr(b*2,b*4); s.d4=cbr(b*4,b*8)
        s.pool=nn.MaxPool2d(2); s.up=lambda: nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False)
        s.u3=cbr(b*8+b*4,b*4); s.u2=cbr(b*4+b*2,b*2); s.u1=cbr(b*2+b,b); s.out=nn.Conv2d(b,1,1)
        s.upl=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False)
    def forward(s,x):
        x1=s.d1(x); x2=s.d2(s.pool(x1)); x3=s.d3(s.pool(x2)); x4=s.d4(s.pool(x3))
        y=s.u3(torch.cat([s.upl(x4),x3],1)); y=s.u2(torch.cat([s.upl(y),x2],1)); y=s.u1(torch.cat([s.upl(y),x1],1))
        return s.out(y)
net=UNet().to(dev); opt=torch.optim.AdamW(net.parameters(),1e-3,weight_decay=1e-4)
lossf=nn.BCEWithLogitsLoss()
CROP=256
def sample():
    i=int(np.random.choice(sel)); x0,y0,x1,y1=BOX[i]
    # random 256 crop inside focus box (clamp)
    cx=np.random.randint(x0,max(x0+1,x1-CROP+1)); cy=np.random.randint(y0,max(y0+1,y1-CROP+1))
    cx=min(cx,IW-CROP); cy=min(cy,IH-CROP)
    r=std(RAW[i])[cy:cy+CROP,cx:cx+CROP]; t=EU[i][cy:cy+CROP,cx:cx+CROP].astype(np.float32)/255.
    if np.random.rand()<.5: r=r[:,::-1]; t=t[:,::-1]
    if np.random.rand()<.5: r=r[::-1]; t=t[::-1]
    k=np.random.randint(4); r=np.rot90(r,k); t=np.rot90(t,k)
    return r.copy(),t.copy()
BS=16
net.train()
for step in range(STEPS):
    xb=np.zeros((BS,1,CROP,CROP),np.float32); tb=np.zeros((BS,1,CROP,CROP),np.float32)
    for b in range(BS): r,t=sample(); xb[b,0]=r; tb[b,0]=t
    x=torch.from_numpy(xb).to(dev); y=torch.from_numpy(tb).to(dev)
    p=net(x); loss=lossf(p,y); opt.zero_grad(); loss.backward(); opt.step()
    if step%500==0: print(f"step{step} loss{loss.item():.4f}",flush=True)
torch.save(net.state_dict(),f"{ROOT}/unet_{TAG}.pt")

# ---- eval on 25 green ----
ann=json.load(open(f"{OUT}/annotations.json")); G=list(ann.keys())
pix_to_um=1000/1296
def areas_of(M): return np.array([r.area*(pix_to_um**2) for r in regionprops(M)])
def CD(a): return 0 if (len(a)==0 or a.sum()==0) else (len(a)*1e6)/a.sum()
def CV(a): return 0 if (len(a)==0 or a.mean()==0) else a.std()/a.mean()
def HEX(M):
    hs=[]
    for c in regionprops(M):
        r0,c0,r1,c1=c.bbox; ms=M[r0:r1,c0:c1].astype(np.float32); ms[ms!=c.label]=0.2; ms[ms==c.label]=1.0
        Ma=c.axis_major_length*0.8; mi=c.axis_minor_length*0.8; a=c.orientation
        hp=np.array([[Ma/2,0],[Ma/4,-mi/2],[-Ma/4,-mi/2],[-Ma/2,0],[-Ma/4,mi/2],[Ma/4,mi/2]])
        R=np.array([[np.cos(a),np.sin(a)],[-np.sin(a),np.cos(a)]]); rh=hp@R
        hm=np.zeros(ms.shape); cy,cx=c.centroid
        th=rh+np.array([np.ceil(cy-r0),np.ceil(cx-c0)]); rr,cc=polygon(th[:,0],th[:,1],hm.shape); hm[rr,cc]=1
        inter=np.logical_and(ms==1.0,hm).sum(); uni=np.logical_or(ms==1.0,hm).sum(); hs.append(inter/uni if uni>0 else 0)
    return np.mean(hs) if hs else 0
def focus_box(img):
    Wb,Hb=538,408; l=cv2.Laplacian(img,cv2.CV_32F,ksize=3); sm=l*l; ii=cv2.integral(sm.astype(np.float64))
    best=-1;bx=by=0
    for y in range(0,IH-Hb+1,16):
        for x in range(0,IW-Wb+1,16):
            s=ii[y+Hb,x+Wb]-ii[y,x+Wb]-ii[y+Hb,x]+ii[y,x]
            if s>best: best=s;bx,by=x,y
    return [bx,by,bx+Wb,by+Hb]
def segment(ep,t=0.2,md=7):
    interior=ep<t; interior=ndi.binary_opening(interior,iterations=1); dist=ndi.distance_transform_edt(interior)
    peaks=peak_local_max(dist,min_distance=md,labels=interior,exclude_border=False)
    mk=np.zeros(ep.shape,np.int32)
    for i,(r,c) in enumerate(peaks,1): mk[r,c]=i
    return watershed(ep,markers=ndi.label(mk>0)[0])
def trim(lab,box,mad_k=3.0):
    x0,y0,x1,y1=[int(v) for v in box]; cand=[]
    for p in regionprops(lab):
        cy,cx=p.centroid
        if not(x0<=cx<x1 and y0<=cy<y1): continue
        if p.bbox[0]<=y0 or p.bbox[2]>=y1 or p.bbox[1]<=x0 or p.bbox[3]>=x1: continue
        if p.area<80 or p.area>4000: continue
        cand.append(p)
    if not cand: return np.zeros_like(lab)
    ar=np.array([p.area for p in cand]); med=np.median(ar); mad=np.median(np.abs(ar-med))+1e-6
    cand=[p for p,a in zip(cand,ar) if abs(a-med)/(1.4826*mad)<mad_k]
    keep=np.zeros_like(lab); n=1
    for p in cand: keep[lab==p.label]=n; n+=1
    return keep
net.eval(); es=[]
with torch.no_grad():
    for ID in G:
        u=readmha(f"{OUT}/mha/{ID}.mha"); xp=np.pad(std(u),((0,PH-IH),(0,0)),mode='reflect')
        x=torch.from_numpy(xp[None,None].astype(np.float32)).to(dev)
        ep=torch.sigmoid(net(x))[0,0,:IH].cpu().numpy()
        rm=trim(segment(ep),focus_box(u)); a=areas_of(rm); g=ann[ID]["gt"]
        es.append([abs(CD(a)-g["CD"])/g["CD"]*100,abs(CV(a)-g["CV"])/g["CV"]*100,abs(HEX(rm)-g["HEX"])/g["HEX"]*100])
es=np.array(es); mu=es.mean(0)
print(f"\n=== U-Net [{TAG}] on 25 green: CD{mu[0]:.1f} CV{mu[1]:.1f} HEX{mu[2]:.1f} OV{mu.mean():.1f} ===",flush=True)
json.dump({"tag":TAG,"thr":THR,"n_train":int(len(sel)),"mpe":mu.tolist(),"ov":float(mu.mean())},
          open(f"{OUT}/unet_{TAG}_results.json","w"),indent=1)
