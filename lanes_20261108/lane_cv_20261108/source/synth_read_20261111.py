"""CV lane (2026-11-11): CV read of the backbone lane's synthetic-pretrained DINOv3-B zero-shot outputs (lane_backbone_20261108/data/
synth_zeroshot_f0pre_20261111.npz: CD_synth, CVarea_synth, HEX_synth; label-free), strictly nested, five folds, against the strict 10.1160.
  (a) corrector: ridge (alpha 100, fixed) on the log residual of the strict 6-component stack (out/search_fixed6_20261111.npz), features
      log CD_s, log CV_s, log HEX_s, log stack CV; fit on folds != f, scale refit MAPE-optimally on the training rows (as seg_vig_cv).
  (b) member: per outer fold a linear calibration log CV ~ log CD_s + log CV_s + log HEX_s fit on folds != f gives an out-of-fold member;
      component = member - log A_cv, added to the 6 components; weights refit by the strict procedure of cv_search_20261111.py."""
import json, sys
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import best_pipeline_20261106 as BP
import stretch_nested_20261103 as N
import ceiling_meta_cpu_20261103 as C
from pathlib import Path
X = Path('/workspace/clear-ec-research/experiments'); LANE = X / 'lane_cv_20261108'
y = BP.y[:, 1]; fold = BP.fold; good = BP.good; ok = y > 0; logH = np.log(BP.H[:, 1]); A = BP._b['A'][:, 1]; GRIDW = np.round(np.arange(0.125, 1.0001, 0.125), 3)
z = np.load(X / 'lane_backbone_20261108/data/synth_zeroshot_f0pre_20261111.npz', allow_pickle=True); pos = {int(i): n for n, i in enumerate(BP.idx)}
S = np.full((len(y), 3), np.nan)
for i, p in zip(z['idx'].astype(int), z['pred']): S[pos[int(i)]] = p
assert np.isfinite(S).all() and (S > 0).all(), 'missing or non-positive synth outputs'
LS = np.log(S.astype(np.float64)); ly = np.log(np.where(ok, y, 1.0))
F6 = np.load(LANE / 'out/search_fixed6_20261111.npz'); P6 = F6['P']                 # strict 6-component OOF CV
ape = lambda p, r: np.abs(p[r] - y[r]) / y[r] * 100
res6 = np.log(P6) - ly
print(json.dumps(dict(strict6=round(float(ape(P6, ok).mean()), 4), r_logCVs_label=round(float(np.corrcoef(LS[ok, 1], ly[ok])[0, 1]), 3),
                      r_logCVs_stack_residual=round(float(np.corrcoef(LS[ok, 1], res6[ok])[0, 1]), 3),
                      r_logCDs_stack_residual=round(float(np.corrcoef(LS[ok, 0], res6[ok])[0, 1]), 3), r_logHEXs_stack_residual=round(float(np.corrcoef(LS[ok, 2], res6[ok])[0, 1]), 3))), flush=True)
# (a) corrector on the strict stack
Pa = P6.copy(); res = ly - np.log(P6)
for f in range(5):
    Xf = np.c_[LS, np.log(P6)]; tr = (fold != f) & good & ok; mu, sd = Xf[tr].mean(0), Xf[tr].std(0) + 1e-9; Z = (Xf - mu) / sd; Am = Z[tr]
    w = np.linalg.solve(Am.T @ Am + 100.0 * np.eye(Am.shape[1]), Am.T @ (res[tr] - res[tr].mean())); d = Z @ w
    q = P6 * np.exp(d); c = C.wmed(y[tr] / q[tr], q[tr] / y[tr]); c0 = C.wmed(y[tr] / P6[tr], P6[tr] / y[tr]); sel = fold == f; Pa[sel] = q[sel] * c / c0
pf = lambda P: [round(float(ape(P, ok & (fold == f)).mean() - ape(P6, ok & (fold == f)).mean()), 4) for f in range(5)]
print(json.dumps(dict(a_corrector=round(float(ape(Pa, ok).mean()), 4), a_delta_vs_strict6=round(float(ape(Pa, ok).mean() - ape(P6, ok).mean()), 4), per_fold=pf(Pa))), flush=True)
# (b) member: out-of-fold calibration, then the strict 7-component refit
mem = np.full(len(y), np.nan)
for f in range(5):
    tr = (fold != f) & good & ok; Xd = np.c_[np.ones(len(y)), LS]; b = np.linalg.lstsq(Xd[tr], ly[tr], rcond=None)[0]; mem[fold == f] = (Xd @ b)[fold == f]
comp_synth = mem - np.log(A)
pool = np.load(LANE / 'out/pool_20261111.npz', allow_pickle=True); names = [str(n) for n in pool['names']]
six = [k for k, n in enumerate(names) if n in ('dino', 'fz:spectrum', 'fz:cov_sqrt', 'fz:autocorr') or n.startswith('graph:tta_s46_tta4_5f+') or n.startswith('graph:gord_s46_tta4_5f+')]
COMP = np.stack([np.stack([pool[f'c{k}_f{f}'] for f in range(5)]) for k in six] + [np.stack([comp_synth] * 5)]); K = len(COMP)
LB = np.stack([logH + BP.BEST[f][:, 1] for f in range(5)])
def stretch_fit(lp, rows):
    r = rows & ok; lb = lp[r]; mu = lb.mean(); yy = y[r]; best = None
    for g in N.GRID:
        q = np.exp(g * (lb - mu) + mu); c = C.wmed(yy / q, q / yy); m = np.mean(np.abs(c * q - yy) / yy)
        if best is None or m < best[0]: best = (m, g, c, mu)
    return best[1:]
def apply_st(lp, prm): g, c, mu = prm; return c * np.exp(g * (lp - mu) + mu)
def pre(gm, w): return LB[gm] + np.tensordot(w, COMP[:, gm, :], axes=1)
def inner(w, f):
    tot = 0.0; n = 0
    for g in range(5):
        if g == f: continue
        lp = pre(g, w); prm = stretch_fit(lp, (fold != g) & (fold != f) & good); sel = (fold == g) & ok; p = apply_st(lp[sel], prm); tot += np.sum(np.abs(p - y[sel]) / y[sel]); n += sel.sum()
    return tot / n * 100
def coord(w, obj, passes=2):
    cur = obj(w)
    for _ in range(passes):
        for k in range(K):
            best = (cur, w[k])
            for gv in np.r_[0.0, GRIDW]:
                if gv == w[k]: continue
                t = w.copy(); t[k] = gv; s = obj(t)
                if s < best[0] - 1e-9: best = (s, gv)
            w[k] = best[1]; cur = best[0]
    return w
Pb = np.full(len(y), np.nan); Wb = {}
for f in range(5):
    w = coord(np.zeros(K), lambda w, f=f: inner(w, f)); lp = pre(f, w); prm = stretch_fit(lp, (fold != f) & good); Pb[fold == f] = apply_st(lp[fold == f], prm); Wb[f] = w.tolist()
print(json.dumps(dict(b_member_alone_r_resid=round(float(np.corrcoef((mem - ly)[ok], res6[ok])[0, 1]), 3), b_member_mape=round(float(np.mean(np.abs(np.exp(mem[ok]) - y[ok]) / y[ok]) * 100), 3),
                      b_stack7=round(float(ape(Pb, ok).mean()), 4), b_delta_vs_strict6=round(float(ape(Pb, ok).mean() - ape(P6, ok).mean()), 4), per_fold=pf(Pb), synth_weight_per_fold=[round(Wb[f][-1], 3) for f in range(5)])), flush=True)
