"""CV lane: fold-0 read of the CV-weighted DINOv3 member against the recipe-loss member (seed 46 both), added to BEST on CV at fixed w 0.125/0.25 (w taken from the earlier cross-fit, not chosen here)."""
import sys
import numpy as np
sys.argv = ['x', '--folds', '4']
src = open('/workspace/clear-ec-research/experiments/lane_cv_20261108/source/cv_stack5_20261108.py').read().split("comps = {}")[0]; exec(src)
r46 = last5('lane_gpu3_20261106/runs/dinov3_b16_f4_s46', 4); rcv = last5('lane_cv_20261108/arms/dinov3_cvw3_f4_s46', 4); assert np.array_equal(r46[0], rcv[0]); r = r46[0]
ref = np.nanmean(E.ape(BP.REF_BEST, y)[fold == 4, 1])
for nm, lg in (('recipe s46', r46[1]), ('cvw3 s46', rcv[1]), ('mean of both', (r46[1] + rcv[1]) / 2)):
    for w in (0.125, 0.25):
        v = np.zeros(N_); v[r] = w * (lg - np.log(A[r, 1])); c = {f: np.zeros((N_, 3)) for f in range(5)}
        for f in range(5): c[f][:, 1] = v
        p = BP.pipe(BP.add(BP.BEST, c)); print('%-14s w %.3f  fold4 dCV %.4f' % (nm, w, np.nanmean(E.ape(p, y)[fold == 4, 1]) - ref), flush=True)
