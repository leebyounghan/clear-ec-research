"""Validate GT-matching ROI recovery on the 25 green images (true annotator boxes known).
For each image: fNLA seg (cached probs) -> per-cell area/hexIoU -> slide median-size windows ->
per-window CD/CV (MAD-trimmed, fnla_seg3 protocol) -> the window best matching GT (CD+CV) is the
"recovered ROI". Measure IoU / center-distance vs the TRUE green box, error at true box, and the
true box's rank in the match ordering. If recovery lands near truth, we can pseudo-label ROIs for
all 8975 unlabeled-train images and train an ROI detector + region-conditioned regression."""
import json, numpy as np, cv2
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import regionprops
from skimage.draw import polygon as skpoly
from scipy import ndimage as ndi
ROOT="datasets/clear_ec_official"; OUT=f"{ROOT}/green_eval"; IH,IW=972,1296; p2u=1000/1296
ann=json.load(open(f"{OUT}/annotations.json")); IDs=list(ann.keys())
boxes=np.array([ann[i]["roi_box_mha"] for i in IDs],float)
BW=int(round(np.median(boxes[:,2]-boxes[:,0]))); BH=int(round(np.median(boxes[:,3]-boxes[:,1])))
print(f"median box {BW}x{BH}",flush=True)
EP=np.load(f"{OUT}/fnla_probs.npz")["edge"]

def cell_hex_iou(ms,c):
    Ma=c.axis_major_length*0.8; mi=c.axis_minor_length*0.8; a=c.orientation
    hp=np.array([[Ma/2,0],[Ma/4,-mi/2],[-Ma/4,-mi/2],[-Ma/2,0],[-Ma/4,mi/2],[Ma/4,mi/2]])
    R=np.array([[np.cos(a),np.sin(a)],[-np.sin(a),np.cos(a)]]); rh=hp@R
    r0,c0,_,_=c.bbox; cy,cx=c.centroid; th=rh+np.array([np.ceil(cy-r0),np.ceil(cx-c0)])
    hm=np.zeros(ms.shape); rr,cc=skpoly(th[:,0],th[:,1],hm.shape); hm[rr,cc]=1
    inter=np.logical_and(ms,hm).sum(); uni=np.logical_or(ms,hm).sum()
    return inter/uni if uni>0 else 0.0

def win_metrics(cy,cx,ar,hx,bb,x0,y0,x1,y1,mad_k=3.0):
    m=(cx>=x0)&(cx<x1)&(cy>=y0)&(cy<y1)&(bb[:,0]>y0)&(bb[:,2]<y1)&(bb[:,1]>x0)&(bb[:,3]<x1)
    if m.sum()<25: return None
    a=ar[m]; h=hx[m]
    med=np.median(a); mad=np.median(np.abs(a-med))+1e-6
    keep=np.abs(a-med)/(1.4826*mad)<mad_k
    a=a[keep]; h=h[keep]
    if len(a)<25: return None
    return len(a)*1e6/a.sum(), a.std()/a.mean(), h.mean(), len(a)

res=[]
for k,ID in enumerate(IDs):
    ep=EP[k]
    interior=ndi.binary_opening(ep<0.2,iterations=1); dist=ndi.distance_transform_edt(interior)
    pk=peak_local_max(dist,min_distance=7,labels=interior,exclude_border=False)
    mk=np.zeros(ep.shape,np.int32)
    for i,(r,c) in enumerate(pk,1): mk[r,c]=i
    lab=watershed(ep,markers=ndi.label(mk>0)[0])
    cy=[];cx=[];ar=[];hx=[];bb=[]
    for c in regionprops(lab):
        if c.area<80 or c.area>4000: continue
        r0,c0,r1,c1=c.bbox
        ms=(lab[r0:r1,c0:c1]==c.label)
        cy.append(c.centroid[0]); cx.append(c.centroid[1]); ar.append(c.area*(p2u**2))
        hx.append(cell_hex_iou(ms,c)); bb.append([r0,c0,r1,c1])
    cy=np.array(cy);cx=np.array(cx);ar=np.array(ar);hx=np.array(hx);bb=np.array(bb)
    g=ann[ID]["gt"]; gt=np.array([g["CD"],g["CV"],g["HEX"]])
    tx0,ty0,tx1,ty1=[int(v) for v in ann[ID]["roi_box_mha"]]
    wins=[]
    for y0 in range(0,IH-BH+1,24):
        for x0 in range(0,IW-BW+1,24):
            wm=win_metrics(cy,cx,ar,hx,bb,x0,y0,x0+BW,y0+BH)
            if wm is None: continue
            cd,cv,hh,n=wm
            score=(abs(cd-gt[0])/gt[0]+abs(cv-gt[1])/gt[1])/2   # match on CD+CV only
            wins.append((score,x0,y0,cd,cv,hh,n))
    if len(wins)<10: print(f"{ID}: too few windows"); continue
    wins.sort()
    sc,bx0,by0,bcd,bcv,bhh,bn=wins[0]
    # true-box metrics (use its own dims)
    tm=win_metrics(cy,cx,ar,hx,bb,tx0,ty0,tx1,ty1)
    def iou(ax0,ay0,ax1,ay1,cx0,cy0,cx1,cy1):
        ix=max(0,min(ax1,cx1)-max(ax0,cx0)); iy=max(0,min(ay1,cy1)-max(ay0,cy0))
        inter=ix*iy; u=(ax1-ax0)*(ay1-ay0)+(cx1-cx0)*(cy1-cy0)-inter
        return inter/u if u>0 else 0
    IoU=iou(bx0,by0,bx0+BW,by0+BH,tx0,ty0,tx1,ty1)
    cdist=float(np.hypot(bx0+BW/2-(tx0+tx1)/2,by0+BH/2-(ty0+ty1)/2))
    # rank of the window closest to the true box
    dists=[np.hypot(x0+BW/2-(tx0+tx1)/2,y0+BH/2-(ty0+ty1)/2) for _,x0,y0,*_ in wins]
    truerank=int(np.argmin(dists)); truerank_pct=truerank/len(wins)*100
    te=None
    if tm: te=[abs(tm[0]-gt[0])/gt[0]*100,abs(tm[1]-gt[1])/gt[1]*100,abs(tm[2]-gt[2])/gt[2]*100]
    re_=[abs(bcd-gt[0])/gt[0]*100,abs(bcv-gt[1])/gt[1]*100,abs(bhh-gt[2])/gt[2]*100]
    res.append(dict(ID=ID,IoU=IoU,cdist=cdist,truerank_pct=truerank_pct,err_true=te,err_rec=re_,nwin=len(wins)))
    print(f"{ID}: IoU {IoU:.2f} cdist {cdist:5.0f}px rank@true {truerank_pct:4.1f}% "
          f"err@true {['%.1f'%v for v in te] if te else 'NA'} err@rec {['%.1f'%v for v in re_]}",flush=True)

I_=[r["IoU"] for r in res]; D=[r["cdist"] for r in res]
ET=np.array([r["err_true"] for r in res if r["err_true"]]); ER=np.array([r["err_rec"] for r in res])
print(f"\n=== SUMMARY n={len(res)} ===")
print(f"recovered-vs-true IoU: median {np.median(I_):.2f}  >0.3: {np.mean(np.array(I_)>0.3)*100:.0f}%  cdist median {np.median(D):.0f}px")
print(f"err at TRUE box   (CD/CV/HEX): {np.median(ET,0).round(2).tolist()} (median)")
print(f"err at RECOVERED  (CD/CV/HEX): {np.median(ER,0).round(2).tolist()} (median, CD/CV fitted=selection-biased)")
json.dump(res,open(f"{OUT}/green_roi_recovery.json","w"))
