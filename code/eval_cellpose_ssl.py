"""Diagnose round-2: eval fine-tuned cpsam_ssl_gated on the 25 green TWO ways:
 (A) full-image segment -> restrict to focus box (as before)
 (B) crop to focus box FIRST -> segment the crop (matches training distribution).
Compare to isolate train/eval mismatch."""
import os, json, numpy as np, cv2, SimpleITK as sitk
from skimage.measure import regionprops
from skimage.draw import polygon
from cellpose import models
ROOT="datasets/clear_ec_official"; OUT=f"{ROOT}/green_eval"; IW,IH=1296,972; pix_to_um=1000/1296
ann=json.load(open(f"{OUT}/annotations.json"))
def readmha(p):
    a=sitk.GetArrayFromImage(sitk.ReadImage(p)).astype(np.float32); return a[0] if a.ndim==3 else a
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
def mpe3(pr,g): return [abs(pr[0]-g["CD"])/g["CD"]*100,abs(pr[1]-g["CV"])/g["CV"]*100,abs(pr[2]-g["HEX"])/g["HEX"]*100]

mdl=models.CellposeModel(gpu=True, pretrained_model=f"{ROOT}/models/cpsam_ssl_gated")
esA=[]; esB=[]; ncA=[]; ncB=[]
for ID in ann:
    u=readmha(f"{OUT}/mha/{ID}.mha"); fb=focus_box(u); g=ann[ID]["gt"]
    MA=mdl.eval(u,batch_size=32)[0].astype(np.int32); rmA=restrict(MA,fb)
    a=areas_of(rmA); esA.append(mpe3((CD(a),CV(a),HEX(rmA)),g)); ncA.append(int(rmA.max()))
    x0,y0,x1,y1=fb; crop=u[y0:y1,x0:x1]
    MB=mdl.eval(crop,batch_size=32)[0].astype(np.int32)
    b=areas_of(MB); esB.append(mpe3((CD(b),CV(b),HEX(MB)),g)); ncB.append(int(MB.max()))
A=np.array(esA).mean(0); B=np.array(esB).mean(0)
print(f"(A) full->restrict : nc~{int(np.mean(ncA))} CD{A[0]:.1f} CV{A[1]:.1f} HEX{A[2]:.1f} OV{A.mean():.1f}")
print(f"(B) box-crop segment: nc~{int(np.mean(ncB))} CD{B[0]:.1f} CV{B[1]:.1f} HEX{B[2]:.1f} OV{B.mean():.1f}")
