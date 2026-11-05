"""Follow-up of the assumption audit (2026-11-05).
  B1 where is the marked region? Box position and marker hull position on the 25; is the ROI really unknown, or nearly fixed?
  B2 what does k (label CD x spacing^2) follow? Adjacent donors and same-year neighbours, by ID distance; partial on predicted CV and HEX, since irregular tilings have a different mean area for the same spacing.
  B3 hex6 as a CV signal: CV and HEX errors of B correlate -0.43 to -0.50, so a topology signal should move both. LOFO blend on CV inside the deployed pipeline.
  B4 best label-free spacing estimate per source (detector peaks, FFT ring), and how much of CD it explains alone, against B."""
import json, re, sys
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import ceiling_meta_cpu_20261103 as C
import abs_tess_20261104 as T
import stretch_nested_20261103 as N
E = C.E; K = list(C.K); b = E.load_base(); y, B, fold, idx = b['y'], b['B'], b['fold'], b['idx']; ids = [re.sub(r'\s+', ' ', i.strip()) for i in b['ids']]; g = E.donors(b['ids']); N_ = len(ids)
fmt = np.array([0 if re.fullmatch(r'\d{4}-\d{2}', i) else 1 for i in ids]); ly = np.log(np.where(y > 0, y, np.nan)); r = ly - np.log(B)
ann = json.load(open('/workspace/clear-ec-research/datasets/clear_ec_official/green_eval/annotations.json')); H_, W_ = 972, 1296
print('=== B1 region position, frame centre', W_ / 2, H_ / 2); rows = []
for n, e in ann.items():
    x0, y0, x1, y1 = e['roi_box_mha']; p = np.asarray(e['markers_mha'], float); rows.append((n, x0, y0, x1, y1, (x0 + x1) / 2, (y0 + y1) / 2, p[:, 0].mean(), p[:, 1].mean(), np.ptp(p[:, 0]), np.ptp(p[:, 1])))
R = np.array([q[1:] for q in rows], float); print('box x0 y0 x1 y1: mean', np.round(R[:, :4].mean(0), 1).tolist(), 'SD', np.round(R[:, :4].std(0), 1).tolist()); print('marker centroid x y: mean', np.round(R[:, 6:8].mean(0), 1).tolist(), 'SD', np.round(R[:, 6:8].std(0), 1).tolist(), 'range x', [float(R[:, 6].min()), float(R[:, 6].max())], 'range y', [float(R[:, 7].min()), float(R[:, 7].max())]); print('marker extent x y: mean', np.round(R[:, 8:10].mean(0), 1).tolist(), 'SD', np.round(R[:, 8:10].std(0), 1).tolist())
PK = T.load_peaks(); sp = np.full(N_, np.nan)
for n in range(N_):
    p = PK.get(int(idx[n]))
    if p is None: continue
    q = p[p[:, 2] >= 0.35, :2]
    if len(q) >= 30: sp[n] = np.median(cKDTree(q).query(q, k=2)[0][:, 1])
z = np.load(C.X / 'meta_ceiling_20261103/sweep/freq_lattice.npz') if (C.X / 'meta_ceiling_20261103/sweep/freq_lattice.npz').exists() else None
print('=== B4 spacing estimators against log CD');
est = {'peaks_t35': np.log(sp)}
if z is not None:
    names = [str(n) for n in z['names']]; F = z['F'][idx].astype(float)
    for nm in ('f0_med', 'f0_top_med', 'ac_r1_med', 'ac_r1_top_med'):
        est[nm] = (-np.log(F[:, names.index(nm)]) if nm.startswith('f0') else np.log(F[:, names.index(nm)]))
for nm, v in est.items():
    o = {}
    for f, s_ in ((0, 'old'), (1, 'new')):
        q = np.isfinite(v) & np.isfinite(ly[:, 0]) & (fmt == f); A = np.polyfit(v[q], ly[q, 0], 1); pr = np.polyval(A, v[q]); o[s_] = dict(pearson=round(float(np.corrcoef(v[q], ly[q, 0])[0, 1]), 3), slope=round(float(A[0]), 2), mape=round(float(np.mean(np.abs(np.exp(pr) - y[q, 0]) / y[q, 0]) * 100), 2), median_log_k=round(float(np.median(ly[q, 0] + 2 * v[q])), 3))
    print(nm, json.dumps(o))
print('B CD MAPE by source', {s_: round(E.official(E.ape(B[fmt == f], y[fmt == f]))['CD'], 2) for f, s_ in ((0, 'old'), (1, 'new'))})
print('=== B2 what k follows'); lk = ly[:, 0] + 2 * np.log(sp); old, new = {}, {}
for n, i in enumerate(ids):
    m = re.fullmatch(r'(\d{4})-(\d{2})', i)
    if m: old[(int(m.group(2)), int(m.group(1)))] = n; continue
    m = re.match(r'(\d{4})-(\d{4})[ -]*([O0][DS])', i)
    if m: new[(int(m.group(1)), int(m.group(2)), m.group(3).replace('0', 'O'))] = n
def cr(P, v):
    P = np.array(P); s = np.isfinite(v[P[:, 0]]) & np.isfinite(v[P[:, 1]]); return round(float(np.corrcoef(v[P[s, 0]], v[P[s, 1]])[0, 1]), 3), int(s.sum())
fell = [(new[k], new[(k[0], k[1], 'OS')]) for k in new if k[2] == 'OD' and (k[0], k[1], 'OS') in new]
lb = np.log(B); adj = lambda v, X: v - X @ np.linalg.lstsq(X[np.isfinite(v)], v[np.isfinite(v)], rcond=None)[0]
X = np.c_[np.ones(N_), lb[:, 1], lb[:, 2], lb[:, 0]]; lk_p = adj(lk, X); print('fellow eyes (new): k', cr(fell, lk), 'k after predicted CV, HEX, CD', cr(fell, lk_p), 'B CD residual', cr(fell, r[:, 0]))
for d in (1, 2, 3, 5, 10, 50):
    print(f'new donors same eye, ID distance {d}: k', cr([(new[k], new[(k[0], k[1] + d, k[2])]) for k in new if (k[0], k[1] + d, k[2]) in new], lk), 'residual', cr([(new[k], new[(k[0], k[1] + d, k[2])]) for k in new if (k[0], k[1] + d, k[2]) in new], r[:, 0]), '| old', f'distance {d}: k', cr([(old[k], old[(k[0], k[1] + d)]) for k in old if (k[0], k[1] + d) in old], lk), 'residual', cr([(old[k], old[(k[0], k[1] + d)]) for k in old if (k[0], k[1] + d) in old], r[:, 0]))
print('=== B3 hex6 on CV, deployed pipeline'); gf = np.load(C.X / 'submission2_search_20261103/green_features.npz', allow_pickle=True); a_, b_, w_ = -0.2084, 0.5989, 0.28; have = gf['f_active10_valid'] & ~gf['seen_leaked'] & np.isfinite(gf['f_active10']); good = ~gf['seen_leaked']; f6 = gf['f_active10'].astype(float)
H = B.copy(); H[have, 2] = np.exp((1 - w_) * np.log(B[have, 2]) + w_ * (a_ + b_ * f6[have]))
def pipeline(base):
    out = np.full_like(base, np.nan)
    for f in range(5): out[fold == f] = N.apply_stretch(base[fold == f], N.fit_stretch(base, y, (fold != f) & good))
    return out
REF = pipeline(H); P = np.full_like(B, np.nan)
for f in range(5):
    tr = (fold != f) & good & have & (y[:, 1] > 0); A = np.polyfit(f6[tr], np.log(y[tr, 1]), 1); best = (9e9, 0)
    for w in np.arange(0, 0.5, 0.02):
        e = np.mean(np.abs(np.exp((1 - w) * np.log(H[tr, 1]) + w * np.polyval(A, f6[tr])) - y[tr, 1]) / y[tr, 1])
        if e < best[0]: best = (e, w)
    C2 = H.copy(); s = have; C2[s, 1] = np.exp((1 - best[1]) * np.log(H[s, 1]) + best[1] * np.polyval(A, f6[s])); st = N.fit_stretch(C2, y, (fold != f) & good); P[fold == f] = N.apply_stretch(C2[fold == f], st); print(' fold', f, 'w', round(float(best[1]), 2), 'slope', round(float(A[0]), 3))
pb = E.paired_bootstrap(E.ape(P, y), E.ape(REF, y), g); s1, s0 = E.official(E.ape(P, y)), E.official(E.ape(REF, y)); print('hex6 on CV too:', round(s1['mean'], 3), 'vs', round(s0['mean'], 3), 'delta', round(pb['delta'], 4), [round(v, 4) for v in pb['ci95']], 'CV', round(s1['CV'] - s0['CV'], 3))
print('done', flush=True)
