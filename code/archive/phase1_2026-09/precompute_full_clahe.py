"""Cache full-resolution (972x1296) CLAHE'd images -> img_full_clahe.dat, aligned to
cache_native_meta_512 ids (so Y matches). Enables full-image-coverage training/eval: GT CV/HEX are
computed over ALL cells, but the model only saw the central 512 (~21% area). Full coverage should
reduce the local->global mismatch on the CV/HEX bottleneck."""
import os, numpy as np, cv2, SimpleITK as sitk, time
ROOT="datasets/clear_ec_official"; H,W=972,1296
meta=np.load(f"{ROOT}/cache_native_meta_512.npz",allow_pickle=True); ids=[str(x) for x in meta["ids"]]; N=int(meta["N"])
dst=np.memmap(f"{ROOT}/img_full_clahe.dat",dtype=np.uint8,mode='w+',shape=(N,H,W))
cl=cv2.createCLAHE(2.0,(8,8)); t0=time.time()
for i,ID in enumerate(ids):
    a=sitk.GetArrayFromImage(sitk.ReadImage(f"{ROOT}/train_mha/{ID}.mha")).astype(np.float32)
    if a.ndim==3: a=a[0]
    a=np.clip(a,0,255).astype(np.uint8)
    if a.shape!=(H,W): a=cv2.resize(a,(W,H))
    dst[i]=cl.apply(a)
    if i%1000==0: print(f"{i}/{N} {time.time()-t0:.0f}s",flush=True)
dst.flush(); print(f"DONE img_full_clahe.dat {time.time()-t0:.0f}s",flush=True)
