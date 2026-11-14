"""D-CDANAT-001 part 3: cheap CD levers on the frozen candidate's strict five-fold OOF (6.1743), each one form fixed before the read, nested:
the correction is fit on the four other folds' strict OOF predictions and applied to the fifth. Not rerun (closed, KG): member-disagreement shift
(exp-cd-uncertainty-shift, +0.012 to +0.025) and robust mask-density variants (exp-cd-mask-density-variants, -0.009 to -0.018 = plain mean, kept).
 (a1) global multiplier: MAPE-optimal (1/y-weighted median of y/p) on the training folds
 (a2) per-predicted-level multiplier: 5 equal-count bins of log prediction (edges from the training folds), one multiplier per bin
 (b)  clip predictions to the training folds' label 0.5 % / 99.5 % quantiles
 (d)  frames with >= 150 interior cells (seg_loose n_int): prediction replaced by the calibrated mask CD (weight 1; calibration on the training folds,
      n_int >= 50 rows, as in the stack)"""
import os, sys, json, numpy as np
os.environ['A2SNAP'] = '1'
import cd_stack_20261109 as K
from cd_common import *
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import best_pipeline_20261106 as BP
z = np.load(X / 'lane_cd_20261108/reads/strict_nesting_20261111.npz'); assert (z['idx'] == BP.idx).all()
p = z['v3_strict']; y = BP.y[:, 0].astype(float); ok = y > 0
def M(q): return float(np.mean(np.abs(q[ok] - y[ok]) / y[ok]) * 100)
def wmed(t, rows): k = rows & ok; return float(C.wmed(y[k] / t[k], t[k] / y[k]))
nint = K.F[:, K.nm['n_int']]
outs = {k: p.copy() for k in ('a1', 'a2', 'b', 'd')}; info = {k: {} for k in outs}
for f in range(5):
    tr = fold != f; te = fold == f
    m = wmed(p, tr); outs['a1'][te] = p[te] * m; info['a1'][f] = round(m, 5)
    e = np.percentile(np.log(p[tr]), [0, 20, 40, 60, 80, 100]); e[0], e[-1] = -np.inf, np.inf; b = np.digitize(np.log(p), e[1:-1])
    ms = [wmed(p, tr & (b == k)) for k in range(5)]; outs['a2'][te] = p[te] * np.array(ms)[b[te]]; info['a2'][f] = [round(v, 4) for v in ms]
    lo, hi = np.percentile(y[tr & ok], [0.5, 99.5]); outs['b'][te] = np.clip(p[te], lo, hi); info['b'][f] = (round(float(lo), 0), round(float(hi), 0), int(((p[te] < lo) | (p[te] > hi)).sum()))
    trm = tr & K.have & ok & good; co = np.polyfit(K.lm[trm], K.ly[trm], 1); Mc = np.exp(np.polyval(co, K.lm)); sel = te & (nint >= 150)
    outs['d'][sel] = Mc[sel]; info['d'][f] = int(sel.sum())
base = M(p); res = {'baseline_strict': round(base, 4)}
for k, q in outs.items():
    res[k] = dict(cd=round(M(q), 4), delta=round(M(q) - base, 4), folds={f: round(float(np.mean(np.abs(q[(fold == f) & ok] - y[(fold == f) & ok]) / y[(fold == f) & ok]) * 100 - np.mean(np.abs(p[(fold == f) & ok] - y[(fold == f) & ok]) / y[(fold == f) & ok]) * 100), 4) for f in range(5)}, fit=info[k])
sel = (nint >= 150) & ok
res['d_diagnostic'] = dict(rows_ge150=int(sel.sum()), stack_mape_on_them=round(float(np.mean(np.abs(p[sel] - y[sel]) / y[sel]) * 100), 3))
print(json.dumps(res, indent=1)); json.dump(res, open(X / 'lane_cd_20261108/reads/cd_levers_20261111.json', 'w'), indent=1)
