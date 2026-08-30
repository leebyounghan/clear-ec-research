"""SSL round-2: fine-tune Cellpose-SAM on scalar-GATED instance masks (proper instance learning,
beyond edge-distillation). Training pairs = focus-box crops of gated pool images + their
watershed+trim instance masks. Eval on 25 green downstream MPE.
"""
import os, json, numpy as np, torch, cv2, SimpleITK as sitk
from skimage.measure import regionprops, label as sklabel
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.draw import polygon
from scipy import ndimage as ndi
from cellpose import models, train as cptrain
ROOT="datasets/clear_ec_official"; OUT=f"{ROOT}/green_eval"
IW,IH=1296,972; PH=976; pix_to_um=1000/1296
THR=float(os.environ.get("THR","12")); EP=int(os.environ.get("EPOCHS","100")); BS=int(os.environ.get("BS","2"))
m=np.load(f"{ROOT}/pool_meta.npz",allow_pickle=True); ids=list(m["ids"]); N=int(m["N"]); mpe=m["mpe"]; BOX=m["box"]
EU=np.memmap(f"{ROOT}/pool_edge_u8.dat",dtype=np.uint8,mode='r',shape=(N,IH,IW))
sel=np.where(mpe.max(1)<=THR)[0]; print(f"gated<= {THR}: {len(sel)} imgs",flush=True)
MDIR=f"{ROOT}/train_mha"
def readmha(p):
    a=sitk.GetArrayFromImage(sitk.ReadImage(p)).astype(np.float32); return a[0] if a.ndim==3 else a
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
# build training pairs = focus-box crops
timg=[]; tlab=[]
for i in sel:
    u=readmha(f"{MDIR}/{ids[i]}.mha"); ep=EU[i].astype(np.float32)/255.
    lab=trim(segment(ep),BOX[i]); x0,y0,x1,y1=[int(v) for v in BOX[i]]
    ic=u[y0:y1,x0:x1]; mc=sklabel(lab[y0:y1,x0:x1])
    if mc.max()>=5: timg.append(ic.astype(np.float32)); tlab.append(mc.astype(np.int32))
print(f"training pairs: {len(timg)} (focus-box crops)",flush=True)

base=models.CellposeModel(gpu=True, pretrained_model="cpsam_models/cpsam_v2")
cptrain.train_seg(base.net, train_data=timg, train_labels=tlab,
    batch_size=BS, learning_rate=1e-5, n_epochs=EP, weight_decay=0.1, normalize=True,
    rescale=False, bsize=256, min_train_masks=5, save_path=ROOT, model_name="cpsam_ssl_gated")
print("training done",flush=True)

# ---- eval on 25 green ----
ann=json.load(open(f"{OUT}/annotations.json"))
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
def restrict(M,box):
    x0,y0,x1,y1=[int(v) for v in box]; keep=np.zeros_like(M); n=1
    for p in regionprops(M):
        cy,cx=p.centroid
        if x0<=cx<x1 and y0<=cy<y1: keep[M==p.label]=n; n+=1
    return keep
es=[]
for ID in ann:
    u=readmha(f"{OUT}/mha/{ID}.mha")
    M=base.eval(u, batch_size=32)[0].astype(np.int32)
    rm=restrict(M, focus_box(u)); a=areas_of(rm); g=ann[ID]["gt"]
    es.append([abs(CD(a)-g["CD"])/g["CD"]*100,abs(CV(a)-g["CV"])/g["CV"]*100,abs(HEX(rm)-g["HEX"])/g["HEX"]*100])
es=np.array(es); mu=es.mean(0)
print(f"\n=== Cellpose-SSL[gated] on 25 green: CD{mu[0]:.1f} CV{mu[1]:.1f} HEX{mu[2]:.1f} OV{mu.mean():.1f} ===",flush=True)
json.dump({"n_train":len(timg),"mpe":mu.tolist(),"ov":float(mu.mean())},open(f"{OUT}/cpsam_ssl_results.json","w"),indent=1)
