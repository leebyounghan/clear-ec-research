"""Domain-randomization augmentation of the pseudo-mask pairs (25 green + gold 9000).
Simulate the messy 9000 (blur, low contrast, brightness, noise, resolution loss, gamma) so cpsam
trained on these generalizes to low-quality images (fixes nc=0 transfer failure). Masks unchanged
(photometric-only). Saves NAUG variants/image -> sam_pseudomasks_aug/{images,masks}."""
import os, numpy as np, cv2, tifffile
from glob import glob
ROOT="datasets/clear_ec_official"
SRCS=[f"{ROOT}/sam_pseudomasks", f"{ROOT}/sam_pseudomasks_9000"]
OUTD=os.environ.get("OUTD",f"{ROOT}/sam_pseudomasks_aug"); NAUG=int(os.environ.get("NAUG","10"))
os.makedirs(f"{OUTD}/images",exist_ok=True); os.makedirs(f"{OUTD}/masks",exist_ok=True)
rng=np.random.default_rng(0)
def degrade(im):
    im=im.astype(np.float32)
    # brightness/contrast toward mean
    mu=im.mean(); c=rng.uniform(0.45,1.15); b=rng.uniform(-25,25); im=(im-mu)*c+mu+b
    # gamma
    im=np.clip(im,0,255); g=rng.uniform(0.6,1.6); im=255*((im/255.0)**g)
    # blur (defocus)
    if rng.random()<0.8:
        k=rng.choice([3,5,7]); im=cv2.GaussianBlur(im,(k,k),rng.uniform(0.4,2.0))
    # resolution loss: down then up
    if rng.random()<0.6:
        f=rng.uniform(0.5,0.9); h,w=im.shape
        im=cv2.resize(cv2.resize(im,(int(w*f),int(h*f)),interpolation=cv2.INTER_AREA),(w,h),interpolation=cv2.INTER_LINEAR)
    # noise
    im=im+rng.normal(0,rng.uniform(2,10),im.shape)
    im=np.clip(im,0,255)
    # re-normalize like the training pipeline (CLAHE was already applied upstream; just minmax)
    return cv2.normalize(im,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
n=0
for SRC in SRCS:
    for ip in sorted(glob(f"{SRC}/images/*.tif")):
        ID=os.path.basename(ip)[:-4]; mp=f"{SRC}/masks/{ID}.tif"
        if not os.path.exists(mp): continue
        im=tifffile.imread(ip); lb=tifffile.imread(mp)
        tag=os.path.basename(SRC)
        # keep 1 clean copy + NAUG degraded
        tifffile.imwrite(f"{OUTD}/images/{tag}_{ID}_c.tif",im); tifffile.imwrite(f"{OUTD}/masks/{tag}_{ID}_c.tif",lb.astype(np.int32)); n+=1
        for a in range(NAUG):
            tifffile.imwrite(f"{OUTD}/images/{tag}_{ID}_{a}.tif",degrade(im)); tifffile.imwrite(f"{OUTD}/masks/{tag}_{ID}_{a}.tif",lb.astype(np.int32)); n+=1
print(f"DONE wrote {n} augmented pairs -> {OUTD}",flush=True)
