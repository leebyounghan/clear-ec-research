"""frame_aug_stats.npz: the per-frame values the measured augmentation reads, plus the noise-versus-intensity exponent."""
import re, json
import numpy as np
from pathlib import Path
LANE = Path('experiments/lane_gpu0_20261106')
z = np.load(LANE/'frame_stats.npz', allow_pickle=True)
ids, names, S, LP, freq = z['ids'].astype(str), list(z['names'].astype(str)), z['stats'].astype(np.float64), z['log_power'].astype(np.float64), z['freq']
c = {n:i for i,n in enumerate(names)}
SW2 = (np.hanning(972)**2).sum()**2
FL = (freq>=0.27)&(freq<=0.40)
P = np.exp(LP); fl = P[:,FL].mean(1,keepdims=True)
sigma = np.sqrt(fl[:,0]/SW2)
m = (freq>=0.07)&(freq<=0.14); x = freq[m]**2; xc = x-x.mean()
focus = -(np.log(np.maximum(P-fl,1e-2*fl))[:,m]*xc).sum(1)/(xc**2).sum()/(4*np.pi**2)
src = np.array([0 if re.fullmatch(r'\d+-\d\d', i.strip()) else 1 for i in ids])
np.savez(LANE/'frame_aug_stats.npz', ids=ids, src=src, sigma=sigma.astype(np.float32), focus=focus.astype(np.float32), p50=S[:,c['p50']].astype(np.float32), p99=S[:,c['p99']].astype(np.float32), mean=S[:,c['mean']].astype(np.float32))
Q=[1,5,25,50,75,95,99]
for s,n in ((0,'A'),(1,'B')):
    k = src==s
    d,mi,b = S[k,c['noise_imm_dark']],S[k,c['noise_imm_mid']],S[k,c['noise_imm_bright']]
    md,mm,mb = S[k,c['mean_dark']],S[k,c['mean_mid']],S[k,c['mean_bright']]
    ok = d>0.5
    kap = np.log(b[ok]/d[ok])/np.log(mb[ok]/np.maximum(md[ok],1))
    print(n,'tertile means',np.median(md).round(1),np.median(mm).round(1),np.median(mb).round(1),'imm',np.median(d).round(2),np.median(mi).round(2),np.median(b).round(2),'kappa pct',np.percentile(kap,Q).round(2).tolist())
    print(n,'sigma',np.percentile(sigma[k],Q).round(2).tolist(),'focus',np.percentile(focus[k],Q).round(2).tolist(),'p50',np.percentile(S[k,c['p50']],Q).tolist(),'p99',np.percentile(S[k,c['p99']],Q).tolist())
    al=(S[k,c['p99']]-S[k,c['p50']]);print(n,'p99-p50',np.percentile(al,Q).tolist())
