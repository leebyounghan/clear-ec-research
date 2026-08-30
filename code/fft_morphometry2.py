"""FFT morphometry v2 (fixes from ring diagnostic): log-magnitude radial profile, running-min
baseline, sharpest-K windows by Laplacian focus, quality gating (band-interior peak + min amp),
parabola refinement, median across windows. alpha calibrated on train. Outputs per-row CD_fft +
relw + amp (+per-image nwin) for stacking/ensembling."""
import numpy as np, cv2, json
from multiprocessing import Pool
from scipy.ndimage import minimum_filter1d, uniform_filter1d
ROOT="datasets/clear_ec_official"; H,W=972,1296; S=512; p2u=1000/1296.
cm=np.load(f"{ROOT}/cache_native_meta_512.npz",allow_pickle=True); Y=cm["Y"].astype(np.float64); N=int(cm["N"])
imgs=np.memmap(f"{ROOT}/img_full_raw.dat",dtype=np.uint8,mode='r',shape=(N,H,W))
perm=np.random.default_rng(42).permutation(N); te,va,tr=perm[:1000],perm[1000:2000],perm[2000:]
_clahe=cv2.createCLAHE(2.0,(8,8)); hann=np.outer(np.hanning(S),np.hanning(S)).astype(np.float32)
FLO,FHI=9,60
def bump_profile(win):
    u=_clahe.apply(np.ascontiguousarray(win)).astype(np.float32)
    u=(u-u.mean())/(u.std()+1e-6)
    F=np.abs(np.fft.fftshift(np.fft.fft2(u*hann)))
    yy,xx=np.mgrid[0:S,0:S]; rr=np.hypot(yy-S/2,xx-S/2).astype(np.int32)
    prof=np.bincount(rr.ravel(),np.log(F+1e-9).ravel(),minlength=S)/np.maximum(np.bincount(rr.ravel(),minlength=S),1)
    prof=prof[:120]
    base=uniform_filter1d(minimum_filter1d(prof,15),15)
    return prof-base
def peak_of(bump):
    k=int(np.argmax(bump[FLO:FHI]))+FLO
    if k<=FLO or k>=FHI-1: return None
    a,b,c=bump[k-1],bump[k],bump[k+1]
    if b<0.35: return None                     # weak ring -> reject window
    denom=a-2*b+c; dk=0.5*(a-c)/denom if abs(denom)>1e-9 else 0.0
    f=k+np.clip(dk,-1,1)
    hh=b/2.0
    lo=k
    while lo>FLO and bump[lo]>hh: lo-=1
    hi=k
    while hi<FHI-1 and bump[hi]>hh: hi+=1
    return f,(hi-lo)/f,b
def one(i):
    im=np.asarray(imgs[i])
    wins=[]
    for r in range(0,H-S+1,115):
        for c in range(0,W-S+1,130):
            wins.append((cv2.Laplacian(im[r:r+S,c:c+S],cv2.CV_32F).var(),r,c))
    wins.sort(reverse=True)
    fs=[];ws=[];amps=[]
    for v,r,c in wins[:4]:                      # 4 sharpest windows
        pk=peak_of(bump_profile(im[r:r+S,c:c+S]))
        if pk is None: continue
        f,relw,amp=pk; fs.append(f); ws.append(relw); amps.append(amp)
    if not fs: return None
    return dict(i=int(i),f=float(np.median(fs)),relw=float(np.median(ws)),
                amp=float(np.median(amps)),nwin=len(fs))
if __name__=="__main__":
    idx=np.concatenate([tr,va,te])
    with Pool(10) as p: res=[r for r in p.map(one,idx) if r]
    D={r["i"]:r for r in res}
    print(f"rings found {len(res)}/{len(idx)}",flush=True)
    # calibrate: log CD = 2*log f + b  (alpha in log space) on TRAIN, robust (median of b)
    trs=[D[i] for i in tr if i in D and Y[i,0]>0]
    b=np.median([np.log(Y[r["i"],0])-2*np.log(r["f"]/(S*p2u)*1000.0) for r in trs])
    def cd_of(r): return float(np.exp(2*np.log(r["f"]/(S*p2u)*1000.0)+b))
    for sp,ix in [("tr",tr),("va",va),("te",te)]:
        rows=[D[i] for i in ix if i in D]
        cd=np.array([cd_of(r) for r in rows]); g=np.array([Y[r["i"]] for r in rows])
        ok=g[:,0]>0
        e=np.abs(cd[ok]-g[ok,0])/g[ok,0]*100
        cor=np.corrcoef(cd[ok],g[ok,0])[0,1]
        cor_cv=np.corrcoef([r["relw"] for r in rows],g[:,1])[0,1]
        cor_hex=np.corrcoef([r["amp"] for r in rows],g[:,2])[0,1]
        print(f"[{sp}] n={len(rows)} CD_fft MAPE {e.mean():.2f} (med {np.median(e):.2f}, corr {cor:.3f}) "
              f"| corr relw-CV {cor_cv:.3f} amp-HEX {cor_hex:.3f}",flush=True)
    out={str(r["i"]):dict(cd=cd_of(r),relw=r["relw"],amp=r["amp"],nwin=r["nwin"]) for r in res}
    json.dump({"b":float(b),"rows":out},open(f"{ROOT}/fft_morphometry2.json","w"))
    print("saved -> fft_morphometry2.json",flush=True)
