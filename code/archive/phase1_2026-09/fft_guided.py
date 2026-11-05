"""CNN-guided FFT ring refinement. Band = +-25% around the expected ring freq derived from a CD prior:
train rows -> GT CD (calibration only), val/test rows -> s2 CNN prediction (member_preds.npz).
If the ring peak within the band is strong, CD_fft = exp(2 log f + b); else row skipped.
Saves fft_guided.json. The blend test then measures val->test transfer of (1-w)*ens + w*fft_cd."""
import numpy as np, cv2, json
from multiprocessing import Pool
from scipy.ndimage import minimum_filter1d, uniform_filter1d
ROOT="datasets/clear_ec_official"; H,W=972,1296; S=512; p2u=1000/1296.
cm=np.load(f"{ROOT}/cache_native_meta_512.npz",allow_pickle=True); Y=cm["Y"].astype(np.float64); N=int(cm["N"])
imgs=np.memmap(f"{ROOT}/img_full_raw.dat",dtype=np.uint8,mode='r',shape=(N,H,W))
perm=np.random.default_rng(42).permutation(N); te,va,tr=perm[:1000],perm[1000:2000],perm[2000:]
z=np.load(f"{ROOT}/member_preds.npz",allow_pickle=True)
prior={}
for i in tr:
    if Y[i,0]>0: prior[int(i)]=Y[i,0]
for arr_i,arr_p in [(z["va_idx"],z["s2_va"]),(z["te_idx"],z["s2_te"])]:
    for i,p in zip(arr_i,arr_p): prior[int(i)]=float(p[0])
# fft2 calibration constant b: log CD = 2 log(fmm) + b  (recompute here on train with guided band)
_clahe=cv2.createCLAHE(2.0,(8,8)); hann=np.outer(np.hanning(S),np.hanning(S)).astype(np.float32)
def bump_profile(win):
    u=_clahe.apply(np.ascontiguousarray(win)).astype(np.float32)
    u=(u-u.mean())/(u.std()+1e-6)
    F=np.abs(np.fft.fftshift(np.fft.fft2(u*hann)))
    yy,xx=np.mgrid[0:S,0:S]; rr=np.hypot(yy-S/2,xx-S/2).astype(np.int32)
    prof=np.bincount(rr.ravel(),np.log(F+1e-9).ravel(),minlength=S)/np.maximum(np.bincount(rr.ravel(),minlength=S),1)
    prof=prof[:120]
    return prof-uniform_filter1d(minimum_filter1d(prof,15),15)
B0=json.load(open(f"{ROOT}/fft_morphometry2.json"))["b"]
def fexp_of(cd): return np.sqrt(np.exp(np.log(np.clip(cd,300,6000))-B0))*(S*p2u)/1000.0
def one(i):
    cd0=prior.get(int(i))
    if cd0 is None: return None
    fe=fexp_of(cd0); lo=max(8,int(fe*0.75)); hi=min(100,int(fe*1.3)+1)
    im=np.asarray(imgs[i])
    wins=[]
    for r in range(0,H-S+1,115):
        for c in range(0,W-S+1,130):
            wins.append((cv2.Laplacian(im[r:r+S,c:c+S],cv2.CV_32F).var(),r,c))
    wins.sort(reverse=True)
    fs=[];amps=[]
    for v,r,c in wins[:4]:
        bump=bump_profile(im[r:r+S,c:c+S])
        k=int(np.argmax(bump[lo:hi]))+lo
        if k<=lo or k>=hi-1: continue
        a,b_,c_=bump[k-1],bump[k],bump[k+1]
        if b_<0.35: continue
        den=a-2*b_+c_; dk=0.5*(a-c_)/den if abs(den)>1e-9 else 0.0
        fs.append(k+np.clip(dk,-1,1)); amps.append(b_)
    if not fs: return None
    return int(i),float(np.median(fs)),float(np.median(amps))
if __name__=="__main__":
    idx=[i for i in np.concatenate([tr,va,te]) if int(i) in prior]
    with Pool(10) as p: res=[r for r in p.map(one,idx) if r]
    D={i:(f,a) for i,f,a in res}
    trs=[i for i in tr if int(i) in D and Y[i,0]>0]
    b=np.median([np.log(Y[i,0])-2*np.log(D[int(i)][0]/(S*p2u)*1000.0) for i in trs])
    def cd_of(i): return float(np.exp(2*np.log(D[int(i)][0]/(S*p2u)*1000.0)+b))
    for sp,ix in [("va",va),("te",te)]:
        rows=[i for i in ix if int(i) in D and Y[i,0]>0]
        cd=np.array([cd_of(i) for i in rows]); g=Y[rows,0]
        e=np.abs(cd-g)/g*100
        print(f"[{sp}] n={len(rows)}/{len(ix)} guided CD_fft MAPE {e.mean():.2f} med {np.median(e):.2f} corr {np.corrcoef(cd,g)[0,1]:.3f}",flush=True)
    json.dump({"b":float(b),"rows":{str(i):dict(cd=cd_of(i),amp=D[int(i)][1]) for i in idx if int(i) in D}},
              open(f"{ROOT}/fft_guided.json","w"))
    print("saved -> fft_guided.json",flush=True)
