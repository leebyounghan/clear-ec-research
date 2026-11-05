"""Per-row tail of the bf16 -> fp16 perturbation of the mean-log-hex6 feature (CPU only).
Reuses the loaders of green_t4_numerics_20261101.py; caches per-family features to features_cache.npz.
Run from /workspace/clear-ec-research:  python -u experiments/green_track_20261028/t4_numerics/feature_tail.py
"""
import sys, json, numpy as np
sys.path.insert(0, 'experiments/green_track_20261028/source')
import green_t4_numerics_20261101 as M
P, OUT, ROOT = M.P, M.OUT, M.ROOT
y, tta4, baseA, fold, seen, keys, F, aa = P.load()
ok = ~seen; n = len(y)
bf, fp = {}, {}
for fam in M.ACTIVE_ORDER:
    kind, src = M.BF16_SOURCES[fam]
    H = np.full(n, np.nan)
    for f in range(5):
        Hf = M.hex6_from(kind, src(f), n); m = np.isfinite(Hf); H[m] = Hf[m]
    bf[fam] = H
    if fam in M.FAMILIES:
        k2 = 'maps' if M.FAMILIES[fam]['kind'] == 's8' else 'peaks'
        H2 = np.full(n, np.nan)
        for f in range(5):
            p = OUT/'fp16'/fam/f'fold{f}'/(f'fold{f}_maps.npz' if k2 == 'maps' else f'fold{f}_sg2_peaks.npz')
            Hf = M.hex6_from(k2, p, n); m = np.isfinite(Hf); H2[m] = Hf[m]
        fp[fam] = H2
    print('loaded', fam, flush=True)
np.savez_compressed(OUT/'features_cache.npz', keys=np.asarray(keys), fold=fold, ok=ok,
                    **{f'bf16_{k}': v for k, v in bf.items()}, **{f'fp16_{k}': v for k, v in fp.items()})

def mean_log(fams, use_fp):
    A = np.log(np.stack([fp[k] if (use_fp and k in fp) else bf[k] for k in fams]))
    with np.errstate(all='ignore'):
        return np.nanmean(A, 0)

params = json.loads((ROOT/'deploy'/'hex6_blend_params.json').read_text())
wb = params['w']*params['b']
res = {}
for name, fams in (('seven', list(fp)), ('ten_mixed', M.ACTIVE_ORDER)):
    a_, b_ = mean_log(fams, False), mean_log(fams, True)
    m = ok & np.isfinite(a_) & np.isfinite(b_)
    d = np.abs(a_[m]-b_[m]); rel = np.abs(np.exp(wb*(b_[m]-a_[m]))-1)
    q = lambda v, p: float(np.quantile(v, p))
    res[name] = dict(n=int(m.sum()), q50=q(d, .5), q90=q(d, .9), q95=q(d, .95), q99=q(d, .99), max=float(d.max()),
                     frac_gt_0p02=float((d > 0.02).mean()), frac_gt_0p03=float((d > 0.03).mean()),
                     frac_rel_hex_gt_0p005=float((rel > 0.005).mean()), rel_hex_q99=q(rel, .99), rel_hex_max=float(rel.max()),
                     p_any_of_25_fails_tol_feat_0p02=float(1-(1-(d > 0.02).mean())**25),
                     p_any_of_25_fails_tol_hex_0p005=float(1-(1-(rel > 0.005).mean())**25),
                     frac_gt_0p02_per_fold=[float((np.abs(a_[m & (fold == h)]-b_[m & (fold == h)]) > 0.02).mean()) for h in range(5)])
z = np.load(ROOT/'deploy'/'regression_fixture_25_v2.npz', allow_pickle=True)
ids = [str(s) for s in z['ids']]; kk = [str(s) for s in keys]
pos = {k: i for i, k in enumerate(kk)}
a_, b_ = mean_log(M.ACTIVE_ORDER, False), mean_log(M.ACTIVE_ORDER, True)
rows = [pos.get(i) for i in ids]
d25 = [float(abs(a_[r]-b_[r])) if (r is not None and np.isfinite(a_[r]) and np.isfinite(b_[r])) else None for r in rows]
vals = [v for v in d25 if v is not None]
res['fixture25_oof_fold4_abs_diff_ten_mixed'] = dict(ids=ids, abs_diff=d25, n_matched=len(vals), n_gt_0p02=int(sum(v > 0.02 for v in vals)),
                                                    n_gt_0p03=int(sum(v > 0.03 for v in vals)), max=(max(vals) if vals else None))
res['params'] = params
res['note'] = ('bf16 vs fp16 OOF mean-log-hex6 feature on fold-clean rows; ten_mixed keeps s8_s47/s8_s48/s8multi_s47 at bf16 '
               '(no fold-clean checkpoints). The 25 fixture rows use fold-4 OOF detectors, not the deployed fold-99 ones: indicative only.')
json.dump(res, open(OUT/'feature_tail.json', 'w'), indent=1)
print(json.dumps(res, indent=1))
