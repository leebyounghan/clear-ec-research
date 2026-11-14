"""CV lane: per-fold mean and SD of each CV component (log member - log reference) and of the label residual of BEST, to see whether fold 1 carries an offset."""
import sys
import numpy as np
sys.argv = ['x', '--folds', '0124', 'dino', 'fz:spectrum', 'fz:cov_sqrt', 'fz:autocorr']
exec(open('/workspace/clear-ec-research/experiments/lane_cv_20261108/source/cv_stack5_20261108.py').read().split("names = list(comps)")[0])
ly = np.log(np.where(y[:, 1] > 0, y[:, 1], np.nan)); res = ly - np.log(BP.REF_BEST[:, 1])
for f in FOLDS:
    m = fold == f
    print('fold', f, 'resid mean %.4f' % np.nanmean(res[m]), ' '.join('%s mean %.4f sd %.4f r_res %.3f' % (k, c[f][m].mean(), c[f][m].std(), np.corrcoef(c[f][m][np.isfinite(res[m])], res[m][np.isfinite(res[m])])[0, 1]) for k, c in comps.items()), flush=True)
