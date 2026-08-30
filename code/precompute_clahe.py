"""Precompute CLAHE(2.0,8x8) on each full 512-cached image once -> img_clahe_512.dat (uint8).
For CROP=512 training this is IDENTICAL to per-crop CLAHE (crop==full image), but removes the
per-step CLAHE CPU cost (711s/epoch -> ~100s/epoch). Train reads this memmap with CLAHE=0."""
import numpy as np, cv2, time
ROOT="datasets/clear_ec_official"; SZ=512
meta=np.load(f"{ROOT}/cache_native_meta_{SZ}.npz",allow_pickle=True); N=int(meta["N"])
src=np.memmap(f"{ROOT}/img_native_{SZ}.dat",dtype=np.uint8,mode='r',shape=(N,SZ,SZ))
dst=np.memmap(f"{ROOT}/img_clahe_{SZ}.dat",dtype=np.uint8,mode='w+',shape=(N,SZ,SZ))
cl=cv2.createCLAHE(2.0,(8,8)); t0=time.time()
for i in range(N):
    dst[i]=cl.apply(np.ascontiguousarray(src[i]))
    if i%1000==0: print(f"{i}/{N} {time.time()-t0:.0f}s",flush=True)
dst.flush(); print(f"DONE img_clahe_{SZ}.dat  {time.time()-t0:.0f}s",flush=True)
