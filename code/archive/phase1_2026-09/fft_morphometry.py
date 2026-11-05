"""Segmentation-free FFT morphometry (Selig 2015 modeRMrec + Homan 2020 peak-width CV proxy).
Cell mosaic -> ring in |FFT| at f' with density = f'^2/alpha (alpha~1.0-1.155). Steps per window:
CLAHE -> hann -> |FFT| -> remove DC peak via morphological reconstruction (Selig) -> radial mean
profile -> parabola-fit peak -> CD_fft = f'^2/alpha ; CV proxy = relative ring width (FWHM/f').
Windows: 5 x 512 crops (C5, same as regression) -> median across windows.
Eval vs GT on val rows; report MAPE for CD_fft and correlation of proxies with GT CV/HEX.
alpha + proxy->CV mapping calibrated on TRAIN rows only (least squares in log space)."""
import numpy as np, cv2, json
from multiprocessing import Pool
from skimage.morphology import reconstruction
ROOT="datasets/clear_ec_official"; H,W=972,1296; PX_MM=1296/1000.  # px per mm... p2u=1000/1296 um/px
cm=np.load(f"{ROOT}/cache_native_meta_512.npz",allow_pickle=True); Y=cm["Y"].astype(np.float64); N=int(cm["N"])
imgs=np.memmap(f"{ROOT}/img_full_raw.dat",dtype=np.uint8,mode='r',shape=(N,H,W))
perm=np.random.default_rng(42).permutation(N); te,va,tr=perm[:1000],perm[1000:2000],perm[2000:]
S=512; hann=np.outer(np.hanning(S),np.hanning(S)).astype(np.float32)
_clahe=cv2.createCLAHE(2.0,(8,8))
p2u=1000/1296.  # um/px
def radial_profile(win):
    u=_clahe.apply(np.ascontiguousarray(win)).astype(np.float32)
    u=(u-u.mean())/(u.std()+1e-6)
    F=np.abs(np.fft.fftshift(np.fft.fft2(u*hann)))
    # Selig: suppress DC/illumination peak by morphological reconstruction (seed = F - h)
    h=F.max()*0.02
    rec=reconstruction(np.clip(F-h,0,None),F,method='dilation')
    P=F-rec  # peaks above reconstructed background... (Selig uses rec to remove low-freq peak)
    yy,xx=np.mgrid[0:S,0:S]; r=np.hypot(yy-S/2,xx-S/2).astype(np.int32)
    prof=np.bincount(r.ravel(),F.ravel(),minlength=S)/np.maximum(np.bincount(r.ravel(),minlength=S),1)
    profP=np.bincount(r.ravel(),P.ravel(),minlength=S)/np.maximum(np.bincount(r.ravel(),minlength=S),1)
    return prof[:S//2],profP[:S//2]
def peak_features(prof, fmin, fmax):
    """parabola-refined peak in [fmin,fmax) + FWHM relative width"""
    seg=prof[fmin:fmax]
    k=int(np.argmax(seg))+fmin
    if k<=fmin or k>=fmax-1: return None
    a,b,c=prof[k-1],prof[k],prof[k+1]
    denom=(a-2*b+c)
    dk=0.5*(a-c)/denom if abs(denom)>1e-9 else 0.0
    f=k+np.clip(dk,-1,1)
    # FWHM around peak on background-subtracted profile
    base=np.percentile(prof[fmin:fmax],20); hh=(b-base)/2+base
    lo=k
    while lo>fmin and prof[lo]>hh: lo-=1
    hi=k
    while hi<fmax-1 and prof[hi]>hh: hi+=1
    return f,(hi-lo)/max(f,1e-6),(b-base)/(abs(base)+1e-6)
C5=[(0,0),(0,S),(H-S,0),(H-S,W-S),(230,392)]
def one(i):
    im=np.asarray(imgs[i])
    fs=[];ws=[];sn=[]
    for r,c in [(0,0),(0,W-S),(H-S,0),(H-S,W-S),((H-S)//2,(W-S)//2)]:
        prof,profP=radial_profile(im[r:r+S,c:c+S])
        # physiological cell pitch 9-45um -> freq bins S*p2u/pitch: 512*0.7716/45=8.8 .. /9=43.9
        pf=peak_features(profP,9,60)
        if pf is None: continue
        f,relw,snr=pf
        fs.append(f); ws.append(relw); sn.append(snr)
    if not fs: return None
    f=np.median(fs); relw=np.median(ws); snr=np.median(sn)
    # density: f cycles per window of S px -> spatial freq f/(S*p2u) per um -> per mm: *1000
    fmm=f/(S*p2u)*1000.0  # cycles per mm
    return dict(i=int(i),fmm=float(fmm),relw=float(relw),snr=float(snr))
if __name__=="__main__":
    idx=np.concatenate([tr,va,te])
    with Pool(10) as p: res=[r for r in p.map(one,idx) if r]
    D={r["i"]:r for r in res}
    print(f"profiles ok {len(res)}/{len(idx)}")
    # calibrate alpha on TRAIN: CD = fmm^2/alpha -> alpha = fmm^2/CD
    trs=[D[i] for i in tr if i in D and Y[i,0]>0]
    al=np.median([r["fmm"]**2/Y[r["i"],0] for r in trs])
    # CV mapping: log CV ~ a*log(relw)+b on train
    X=np.array([[np.log(r["relw"]),1.0] for r in trs]); yv=np.log([Y[r["i"],1] for r in trs])
    coef,_,_,_=np.linalg.lstsq(X,yv,rcond=None)
    out={}
    for sp,ix in [("va",va),("te",te)]:
        rows=[D[i] for i in ix if i in D]
        cd=np.array([r["fmm"]**2/al for r in rows]); g=np.array([Y[r["i"]] for r in rows])
        cv=np.exp(coef[0]*np.log([r["relw"] for r in rows])+coef[1])
        e_cd=np.abs(cd-g[:,0])/g[:,0]*100; e_cv=np.abs(cv-g[:,1])/g[:,1]*100
        cor_cd=np.corrcoef(cd,g[:,0])[0,1]; cor_cv=np.corrcoef([r["relw"] for r in rows],g[:,1])[0,1]
        cor_hex=np.corrcoef([r["snr"] for r in rows],g[:,2])[0,1]
        print(f"[{sp}] n={len(rows)} CD_fft MAPE {e_cd.mean():.2f} (med {np.median(e_cd):.2f}, corr {cor_cd:.3f}) | "
              f"CV_fft MAPE {e_cv.mean():.2f} (corr relw-CV {cor_cv:.3f}) | corr snr-HEX {cor_hex:.3f}",flush=True)
        for r in rows: out[str(r["i"])]=dict(sp=sp,cd=float(r["fmm"]**2/al),relw=r["relw"],snr=r["snr"])
    for r in trs: out[str(r["i"])]=dict(sp="tr",cd=float(r["fmm"]**2/al),relw=r["relw"],snr=r["snr"])
    json.dump({"alpha":float(al),"cv_coef":coef.tolist(),"rows":out},open(f"{ROOT}/fft_morphometry.json","w"))
    print(f"alpha={al:.3f} saved -> fft_morphometry.json",flush=True)
