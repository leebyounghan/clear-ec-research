"""Convert submitted human border PNGs into instance training masks (human-boundary gold).
Each submitted border_<id>.png is the drawn border layer at the DISPLAY resolution (native*UP).
Downscale to native, label connected non-border regions -> instance mask; drop the image-edge
components (partial/background). Pair with the native crop (decoded from server_data.json) -> gold
tif pairs. These carry TRUE cell areas (unlike SAM/cpsam pseudo-masks) -> the fix for CV.
Also prints the mask's CD/CV/HEX vs GT so we see annotation quality before training."""
import os, json, base64, glob, numpy as np, cv2, tifffile
from skimage.measure import label as cclabel
from skimage.segmentation import clear_border
from skimage.measure import regionprops
from scipy.spatial import ConvexHull
ROOT="datasets/clear_ec_official"; p2u=1000/1296
BB=f"{ROOT}/boundary_batch"; SUB=f"{BB}/submitted"; OUT=f"{ROOT}/boundary_gold"; os.makedirs(OUT,exist_ok=True)
data=json.load(open(f"{BB}/server_data.json")); byid={it["id"]:it for it in data["images"]}
def dec(durl):
    b=base64.b64decode(durl.split(",",1)[1]); a=cv2.imdecode(np.frombuffer(b,np.uint8),cv2.IMREAD_UNCHANGED); return a
def CV(a): return 0 if (len(a)<2 or a.mean()==0) else a.std()/a.mean()
def HEX_adj(m):
    lbls=np.unique(m); lbls=lbls[lbls>0]
    if len(lbls)<3: return 0
    adj={int(l):set() for l in lbls}; border=set()
    for ax,sh in [(0,1),(0,-1),(1,1),(1,-1)]:
        r=np.roll(m,sh,axis=ax); d=(m!=r); aa=m[d]; bb=r[d]
        for x,y in zip(aa,bb):
            if x==0 or y==0:
                if x>0: border.add(int(x))
                if y>0: border.add(int(y))
            else: adj[int(x)].add(int(y)); adj[int(y)].add(int(x))
    deg=[len(adj[int(l)]) for l in lbls if int(l) not in border]
    return float(np.mean([1.0 if dd==6 else 0.0 for dd in deg])) if deg else 0
def e(a,b): return abs(a-b)/b*100 if b else 0
subs=sorted(glob.glob(f"{SUB}/border_*.png")); n=0
print(f"{'id':>7} {'ncell':>5} {'CDe':>5} {'CVe':>5} {'HEXe':>5}")
for sp in subs:
    ID=os.path.basename(sp)[7:-4]
    if ID not in byid: print(f"{ID}: not in batch, skip"); continue
    it=byid[ID]; up=it["up"]; nW,nH=it["nativeW"],it["nativeH"]
    bimg=cv2.imread(sp,cv2.IMREAD_UNCHANGED)  # RGBA at display res
    alpha=bimg[:,:,3] if bimg.ndim==3 and bimg.shape[2]==4 else (bimg if bimg.ndim==2 else bimg[:,:,0])
    bnat=cv2.resize(alpha,(nW,nH),interpolation=cv2.INTER_AREA)
    isborder=bnat>40
    lab=cclabel(~isborder, connectivity=1).astype(np.int32)
    lab=clear_border(lab)  # drop partial edge/background comps
    # relabel 1..n, drop tiny
    out=np.zeros_like(lab); k=1
    for rp in regionprops(lab):
        if rp.area<12: continue
        out[lab==rp.label]=k; k+=1
    # crop image from server_data
    crop=dec(it["img"]); crop=cv2.resize(crop if crop.ndim==2 else crop[:,:,0],(nW,nH),interpolation=cv2.INTER_AREA)
    tifffile.imwrite(f"{OUT}/{ID}_img.tif",crop.astype(np.uint8)); tifffile.imwrite(f"{OUT}/{ID}_mask.tif",out.astype(np.int32))
    a=np.array([rp.area*(p2u**2) for rp in regionprops(out)]); nc=int(out.max())
    try: ha=ConvexHull(np.array([rp.centroid for rp in regionprops(out)])[:,::-1]).volume*(p2u**2)/1e6; cd=nc/ha
    except Exception: cd=0
    g=it["gt"]; print(f"{ID:>7} {nc:>5} {e(cd,g['CD']):>4.0f}% {e(CV(a),g['CV']):>4.0f}% {e(HEX_adj(out),g['HEX']):>4.0f}%",flush=True); n+=1
print(f"\nDONE {n} human-boundary gold pairs -> {OUT}",flush=True)
