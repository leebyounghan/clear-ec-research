"""Upper bound of center-dot annotation: marker-seeded watershed on the fNLA edge map.
For each green image, seed watershed with the TRUE annotator markers (perfect dots), segment,
compute CD/CV/HEX at the true box, compare vs (a) GT and (b) the unseeded automatic seg.
If seeded CV/HEX << unseeded, dot-level annotation (which Claude CAN do visually) is worth it;
boundaries come free from edge map + seeds."""
import json, numpy as np
from skimage.segmentation import watershed
from skimage.measure import regionprops
from skimage.draw import polygon as skpoly
from scipy import ndimage as ndi
ROOT="datasets/clear_ec_official"; OUT=f"{ROOT}/green_eval"; p2u=1000/1296
ann=json.load(open(f"{OUT}/annotations.json")); IDs=list(ann.keys())
EP=np.load(f"{OUT}/fnla_probs.npz")["edge"]

def cell_hex_iou(ms,c):
    Ma=c.axis_major_length*0.8; mi=c.axis_minor_length*0.8; a=c.orientation
    hp=np.array([[Ma/2,0],[Ma/4,-mi/2],[-Ma/4,-mi/2],[-Ma/2,0],[-Ma/4,mi/2],[Ma/4,mi/2]])
    R=np.array([[np.cos(a),np.sin(a)],[-np.sin(a),np.cos(a)]]); rh=hp@R
    r0,c0,_,_=c.bbox; cy,cx=c.centroid; th=rh+np.array([np.ceil(cy-r0),np.ceil(cx-c0)])
    hm=np.zeros(ms.shape); rr,cc=skpoly(th[:,0],th[:,1],hm.shape); hm[rr,cc]=1
    inter=np.logical_and(ms,hm).sum(); uni=np.logical_or(ms,hm).sum()
    return inter/uni if uni>0 else 0.0

def metrics_at_box(lab,box,mad_k=3.0,amin=80,amax=4000):
    x0,y0,x1,y1=[int(v) for v in box]; A=[];H=[]
    for p in regionprops(lab):
        cy,cx=p.centroid
        if not(x0<=cx<x1 and y0<=cy<y1): continue
        if p.area<amin or p.area>amax: continue
        r0,c0,r1,c1=p.bbox
        ms=(lab[r0:r1,c0:c1]==p.label)
        A.append(p.area*(p2u**2)); H.append(cell_hex_iou(ms,p))
    if len(A)<10: return None
    A=np.array(A); H=np.array(H)
    med=np.median(A); mad=np.median(np.abs(A-med))+1e-6
    k=np.abs(A-med)/(1.4826*mad)<mad_k; A=A[k]; H=H[k]
    return len(A)*1e6/A.sum(), A.std()/A.mean(), H.mean(), len(A)

E_seed=[];E_auto=[]
for k,ID in enumerate(IDs):
    ep=EP[k]; g=ann[ID]["gt"]; gt=np.array([g["CD"],g["CV"],g["HEX"]])
    box=ann[ID]["roi_box_mha"]
    # marker-seeded: domain = box + margin; true markers seed inside the box, auto-peaks seed
    # the margin ring so box-edge cells face competition instead of flooding outward
    from skimage.feature import peak_local_max
    x0,y0,x1,y1=[int(v) for v in box]; PAD=45
    X0,Y0,X1,Y1=max(0,x0-PAD),max(0,y0-PAD),min(1296,x1+PAD),min(972,y1+PAD)
    dom=np.zeros(ep.shape,bool); dom[Y0:Y1,X0:X1]=True
    mk=np.zeros(ep.shape,np.int32); nid=0
    for (mx,my) in ann[ID]["markers_mha"]:
        r,c=int(round(my)),int(round(mx))
        if 0<=r<972 and 0<=c<1296: nid+=1; mk[r,c]=nid
    interior0=ndi.binary_opening(ep<0.2,iterations=1)
    ring=dom.copy(); ring[y0:y1,x0:x1]=False
    d0=ndi.distance_transform_edt(interior0&ring)
    for (r,c) in peak_local_max(d0,min_distance=7,labels=interior0&ring,exclude_border=False):
        if mk[max(0,r-8):r+9,max(0,c-8):c+9].any(): continue
        nid+=1; mk[r,c]=nid
    labS=watershed(ep,markers=mk,mask=dom)
    mS=metrics_at_box(labS,box,amin=40,amax=6000)
    # unseeded auto seg (same as recovery script)
    from skimage.feature import peak_local_max
    interior=ndi.binary_opening(ep<0.2,iterations=1); dist=ndi.distance_transform_edt(interior)
    pk=peak_local_max(dist,min_distance=7,labels=interior,exclude_border=False)
    mk2=np.zeros(ep.shape,np.int32)
    for i,(r,c) in enumerate(pk,1): mk2[r,c]=i
    labA=watershed(ep,markers=ndi.label(mk2>0)[0])
    mA=metrics_at_box(labA,box)
    if mS is None or mA is None: print(f"{ID}: skip"); continue
    eS=[abs(mS[j]-gt[j])/gt[j]*100 for j in range(3)]
    eA=[abs(mA[j]-gt[j])/gt[j]*100 for j in range(3)]
    E_seed.append(eS); E_auto.append(eA)
    print(f"{ID}: seeded n{mS[3]:4d} CD{eS[0]:5.1f} CV{eS[1]:5.1f} HEX{eS[2]:5.1f} | auto n{mA[3]:4d} CD{eA[0]:5.1f} CV{eA[1]:5.1f} HEX{eA[2]:5.1f}",flush=True)
S=np.array(E_seed);A=np.array(E_auto)
print(f"\n=== n={len(S)} (median err%) ===")
print(f"marker-SEEDED  CD{np.median(S,0)[0]:5.2f} CV{np.median(S,0)[1]:5.2f} HEX{np.median(S,0)[2]:5.2f} | mean {S.mean(0).round(2).tolist()}")
print(f"AUTO unseeded  CD{np.median(A,0)[0]:5.2f} CV{np.median(A,0)[1]:5.2f} HEX{np.median(A,0)[2]:5.2f} | mean {A.mean(0).round(2).tolist()}")
