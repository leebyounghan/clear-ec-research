"""Prep training tiles from the gold annotations: crop each gold (full 972x1296) to its labeled
region bbox (+margin) -> dense cell tile, then domain-randomization augment (blur/contrast/noise/
scale/gamma) so cpsam generalizes to messy test. Saves gold_aug/{images,masks}."""
import os, glob, numpy as np, cv2, tifffile
ROOT="datasets/clear_ec_official"; GOLD=os.environ.get("GOLD",f"{ROOT}/annotation_drafts/gold"); OUT=os.environ.get("OUT",f"{ROOT}/gold_aug"); NAUG=int(os.environ.get("NAUG","18"))
os.makedirs(f"{OUT}/images",exist_ok=True); os.makedirs(f"{OUT}/masks",exist_ok=True)
rng=np.random.default_rng(0)
def degrade(im):
    im=im.astype(np.float32); mu=im.mean(); c=rng.uniform(0.5,1.15); b=rng.uniform(-22,22); im=(im-mu)*c+mu+b
    im=np.clip(im,0,255); im=255*((im/255.0)**rng.uniform(0.6,1.6))
    if rng.random()<0.8: k=rng.choice([3,5,7]); im=cv2.GaussianBlur(im,(k,k),rng.uniform(0.4,2.0))
    if rng.random()<0.6:
        f=rng.uniform(0.55,0.9); h,w=im.shape
        im=cv2.resize(cv2.resize(im,(int(w*f),int(h*f)),interpolation=cv2.INTER_AREA),(w,h),interpolation=cv2.INTER_LINEAR)
    im=im+rng.normal(0,rng.uniform(2,9),im.shape); im=np.clip(im,0,255)
    return cv2.normalize(im,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
n=0
for ip in sorted(glob.glob(f"{GOLD}/*_img.tif")):
    ID=os.path.basename(ip).split("_")[0]; mp=f"{GOLD}/{ID}_mask.tif"
    if not os.path.exists(mp): continue
    img=tifffile.imread(ip); lab=tifffile.imread(mp).astype(np.int32)
    ys,xs=np.where(lab>0)
    if len(ys)<50: continue
    m=24; y0,y1=max(0,ys.min()-m),min(img.shape[0],ys.max()+m); x0,x1=max(0,xs.min()-m),min(img.shape[1],xs.max()+m)
    ic=img[y0:y1,x0:x1]; mc=lab[y0:y1,x0:x1]
    tifffile.imwrite(f"{OUT}/images/{ID}_c.tif",ic.astype(np.uint8)); tifffile.imwrite(f"{OUT}/masks/{ID}_c.tif",mc); n+=1
    for a in range(NAUG):
        tifffile.imwrite(f"{OUT}/images/{ID}_{a}.tif",degrade(ic)); tifffile.imwrite(f"{OUT}/masks/{ID}_{a}.tif",mc); n+=1
print(f"DONE {n} training tiles -> {OUT} (from {len(glob.glob(f'{GOLD}/*_img.tif'))} gold)",flush=True)
