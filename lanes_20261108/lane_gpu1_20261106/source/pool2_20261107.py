"""Two-fold pooled frozen read for the screen gate."""
import json, sys
sys.path.insert(0,'/workspace/clear-ec-research/experiments/lane_gpu1_20261106/source')
import numpy as np, frozen_read_20261106 as F
E=F.E; B=F.B; K=F.K
def two(arms, refs, pool):
    e1,e0,g,per=[],[],[],[]
    for f,a,r in zip((4,0),arms,refs):
        p,rows,_=F.read(a,f,pool); p0,rows0,_=F.read(r,f,pool); assert np.array_equal(rows,rows0)
        x,x0=E.ape(p,B['y'][rows]),E.ape(p0,B['y'][rows]); e1.append(x); e0.append(x0); g.append(F.GP[rows]); per.append(round(E.official(x)['mean']-E.official(x0)['mean'],4))
    e1,e0=np.concatenate(e1),np.concatenate(e0); g=np.unique(np.concatenate(g),return_inverse=True)[1]
    pb=E.paired_bootstrap(e1,e0,g); s,s0=E.official(e1),E.official(e0)
    return dict(pool=pool,delta=round(pb['delta'],4),ci95=[round(v,4) for v in pb['ci95']],fold4_fold0=per,per={k:round(s[k]-s0[k],4) for k in K})
for pool in ('std','mean','both'):
    print('cropopt_both',json.dumps(two(('cropopt_both_f4','cropopt_both_f0'),('ssl5_f4','ssl5_f0'),pool)),flush=True)
