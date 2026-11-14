import sys, types
from pathlib import Path
import numpy as np
sys.argv=['x','make_input']
import recipeA_ssl_promote_20261107 as Pm
R=Pm.R
for f,p in ((1,'experiments/ssl_screen_20260911/preflight/fold1.npz'),(3,'experiments/overnight_target_20260912/preflight/fold3.npz'),(2,'experiments/overnight_target_20260912/preflight/fold2.npz'),(99,'experiments/fulldata_recipeA_20261028/fulldata_regression_split.npz')):
    R.SPLIT=Path(p); ids,y,tr,va=Pm.arrays(); print('fold',f,'train',len(tr),'val',len(va),'zero rows in val',int((y[va]==0).any(1).sum()),'train min',float(y[tr].min()))
y0=np.array([[1053.,.03,0.],[520.,.11,0.]]); p0=np.array([[1000.,.03,.5],[600.,.22,.4]]); print(Pm.official_metrics(p0,y0))
rng=np.random.default_rng(5); y1=rng.uniform(.2,3000.,(64,3)); p1=y1*rng.uniform(.8,1.2,y1.shape); print(Pm.official_metrics(p1,y1)==Pm.FROZEN_METRICS(p1,y1))
