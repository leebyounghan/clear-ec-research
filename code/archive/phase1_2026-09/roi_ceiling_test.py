"""Ceiling test: does feeding the model the TRUE annotator ROI location help CV/HEX?
On the 25 green images (true roi_box_mha known), run the s2 regression on:
  (a) central-512 native crop   (what the container does now)
  (b) 512 native crop centered on the true ROI centroid  (region-correct input)
  (c) the ROI box resized to 512  (region-correct + scale-normalized)
Compare CV/HEX MAPE vs the 25 GTs. If (b)/(c) << (a), ROI localization is the lever; build the detector.
Honest caveat: s2 was trained on central-512 -> ROI-centered/resized inputs are somewhat OOD, so this
UNDER-estimates the achievable gain (a model trained on ROI crops would do better)."""
import json, numpy as np, torch, torch.nn as nn, cv2, SimpleITK as sitk
from cellpose import models
ROOT="datasets/clear_ec_official"; OUT=f"{ROOT}/green_eval"; SZ=512; CROP=256; dev="cuda"
LM=np.array([7.919919967651367,-0.9824771285057068,-0.6211431622505188],np.float32)
LS=np.array([0.18971863389015198,0.18813571333885193,0.2748945653438568],np.float32)
_cl=cv2.createCLAHE(2.0,(8,8)); C5=[(0,0),(0,256),(256,0),(256,256),(128,128)]
def dih(a,k): return np.rot90(a[:,::-1] if k>=4 else a,k%4)
def n99(u): u=_cl.apply(np.clip(u,0,255).astype(np.uint8)).astype(np.float32); lo,hi=np.percentile(u,(1,99)); return np.clip((u-lo)/(hi-lo+1e-6),0,1)
class SN(nn.Module):
    def __init__(s,enc):
        super().__init__(); s.enc=enc; s.pool=nn.AdaptiveAvgPool2d(1); s.ln=nn.LayerNorm(256)
        def h(): return nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Dropout(.2),nn.Linear(128,1))
        s.hCD,s.hCV,s.hHEX=h(),h(),h()
    def forward(s,x): g=s.ln(s.pool(s.enc(x)).flatten(1)); return torch.cat([s.hCD(g),s.hCV(g),s.hHEX(g)],1)
base=models.CellposeModel(gpu=True, pretrained_model="cpsam_models/cpsam_v2")
net=SN(base.net.encoder.float()).to(dev); net.load_state_dict(torch.load(f"{ROOT}/regress_ft_best_clahe_s2.pt"),strict=False); net.eval()
@torch.no_grad()
def predict(win512):
    xs=[]
    for r,c in C5:
        w=n99(win512[r:r+CROP,c:c+CROP])
        for k in range(8): xs.append(np.stack([np.ascontiguousarray(dih(w,k))]*3,0))
    x=torch.from_numpy(np.stack(xs)).to(dev)
    with torch.autocast("cuda",dtype=torch.bfloat16): p=net(x).float().mean(0).cpu().numpy()
    return np.exp(p*LS+LM)
ann=json.load(open(f"{OUT}/annotations.json")); IDs=list(ann.keys())
def crop512(img,cy,cx):
    H,W=img.shape; r0=int(np.clip(cy-SZ//2,0,H-SZ)); c0=int(np.clip(cx-SZ//2,0,W-SZ)); return img[r0:r0+SZ,c0:c0+SZ]
EA=[];EB=[];EC=[]
for ID in IDs:
    u=sitk.GetArrayFromImage(sitk.ReadImage(f"{OUT}/mha/{ID}.mha")).astype(np.uint8)
    if u.ndim==3: u=u[0]
    H,W=u.shape; g=ann[ID]["gt"]; G=np.array([g["CD"],g["CV"],g["HEX"]])
    x0,y0,x1,y1=[int(v) for v in ann[ID]["roi_box_mha"]]; cy,cx=(y0+y1)/2,(x0+x1)/2
    a=predict(crop512(u,H/2,W/2))                       # central
    b=predict(crop512(u,cy,cx))                         # ROI-centered, native
    roi=u[y0:y1,x0:x1]; roi512=cv2.resize(roi,(SZ,SZ),interpolation=cv2.INTER_AREA)  # ROI resized
    c=predict(roi512)
    EA.append(np.abs(a-G)/G*100); EB.append(np.abs(b-G)/G*100); EC.append(np.abs(c-G)/G*100)
EA=np.array(EA).mean(0);EB=np.array(EB).mean(0);EC=np.array(EC).mean(0)
print(f"(a) central-512     CD{EA[0]:5.2f} CV{EA[1]:5.2f} HEX{EA[2]:5.2f} OV{EA.mean():.2f}")
print(f"(b) ROI-centered    CD{EB[0]:5.2f} CV{EB[1]:5.2f} HEX{EB[2]:5.2f} OV{EB.mean():.2f}")
print(f"(c) ROI-resized512  CD{EC[0]:5.2f} CV{EC[1]:5.2f} HEX{EC[2]:5.2f} OV{EC.mean():.2f}")
print(f"\ngain(a->b) CV {EA[1]-EB[1]:+.2f} HEX {EA[2]-EB[2]:+.2f} | gain(a->c) CV {EA[1]-EC[1]:+.2f} HEX {EA[2]-EC[2]:+.2f}")
