"""Should hex6 lean on the detectors that saw no challenge labels (2026-11-05)? Cell graphs on pre-only detectors gain -0.043 to -0.051; on a detector fine-tuned on the 20 annotated frames -0.006. The shipped hex6 averages nine fine-tuned detectors and one pre-only. Whole-frame hex6 per family from stored peaks at 0.35; compositions: shipped ten (as a LOFO refit), pre-only x3, sg2 x5, all eight. p' = exp((1 - w) log B + w (a + b f)), a, b, w on the training folds, then stretch. Gate written before reading: better than the shipped ten refit by <= -0.01 on the mean, CI upper bound < 0, at least 4 of 5 folds negative. Forked pool, clamp(16 - load1 - 1, 2, 12)."""
import json, os, sys, warnings
from multiprocessing import get_context
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import window_multi_20261105 as M
import legible_window_features_20261105 as L
OUT = '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/sweep/hex6_families.npz'; PK = None


def one(r):
    o = []
    for nm in M.FAM:
        pk = PK[nm].get(r)
        if pk is None:
            o.append(np.nan); continue
        q = pk[pk[:, 2] >= 0.35, :2].astype(np.float64); s = L.stats(q) if len(q) >= 20 else [np.nan]; o.append(s[0])
    return r, o


def main():
    global PK
    if not os.path.exists(OUT):
        PK = {k: M.load(v) for k, v in M.FAM.items()}; workers = int(min(12, max(2, 16 - os.getloadavg()[0] - 1))); F = np.full((8954, len(M.FAM)), np.nan, np.float32)
        with get_context('fork').Pool(workers) as pool:
            for r, v in pool.imap_unordered(one, range(8954), chunksize=16): F[r] = v
        np.savez(OUT, F=F, families=np.array(list(M.FAM))); print('features done', workers, flush=True)
    import ceiling_meta_cpu_20261103 as C
    import stretch_nested_20261103 as N
    E = C.E; K = list(C.K); b = E.load_base(); y, B, fold, idx = b['y'], b['B'], b['fold'], b['idx']; g = E.donors(b['ids']); z = np.load(OUT); fam = [str(x) for x in z['families']]; F = z['F'][idx].astype(np.float64); F = np.where(F > 0, np.log(np.where(F > 0, F, 1)), np.nan)
    gf = np.load(C.X / 'submission2_search_20261103/green_features.npz', allow_pickle=True); good = ~gf['seen_leaked']; f10 = gf['f_active10'].astype(float); f10[~gf['f_active10_valid']] = np.nan
    pre = [fam.index(q) for q in fam if q.startswith('preonly')]; sg = [fam.index(q) for q in fam if q.startswith('sg2')]
    ms = lambda c: np.nanmean(F[:, c], 1)
    FEAT = {'shipped ten': f10, 'preonly s46': F[:, fam.index('preonly')], 'preonly s47': F[:, fam.index('preonly_s47')], 'preonly s48': F[:, fam.index('preonly_s48')], 'preonly x3': ms(pre), 'sg2 x5': ms(sg), 'all eight': ms(pre + sg)}
    def run(f_):
        P = np.full_like(B, np.nan); ws = []
        for fo in range(5):
            tr = (fold != fo) & good & np.isfinite(f_) & (y[:, 2] > 0); A = np.polyfit(f_[tr], np.log(y[tr, 2]), 1); best = (9e9, 0.0)
            for w in np.arange(0, 0.71, 0.02):
                e = np.mean(np.abs(np.exp((1 - w) * np.log(B[tr, 2]) + w * np.polyval(A, f_[tr])) - y[tr, 2]) / y[tr, 2])
                if e < best[0] - 1e-7: best = (e, w)
            C_ = B.copy(); s = np.isfinite(f_); C_[s, 2] = np.exp((1 - best[1]) * np.log(B[s, 2]) + best[1] * np.polyval(A, f_[s])); P[fold == fo] = N.apply_stretch(C_[fold == fo], N.fit_stretch(C_, y, (fold != fo) & good)); ws.append(round(float(best[1]), 2))
        return P, ws
    REF, wr = run(FEAT['shipped ten']); print('shipped ten refit', round(E.official(E.ape(REF, y))['mean'], 3), 'w', wr, flush=True); rep = {}
    for nm, f_ in FEAT.items():
        s = np.isfinite(f_) & (y[:, 2] > 0); p, ws = run(f_); pb = E.paired_bootstrap(E.ape(p, y), E.ape(REF, y), g); o = E.official(E.ape(p, y)); per = [round(E.official(E.ape(p[fold == q], y[fold == q]))['mean'] - E.official(E.ape(REF[fold == q], y[fold == q]))['mean'], 3) for q in range(5)]; rep[nm] = dict(r_log_hex=round(float(np.corrcoef(f_[s], np.log(y[s, 2]))[0, 1]), 3), score=round(o['mean'], 3), HEX=round(o['HEX'], 3), w=ws, delta=round(pb['delta'], 4), ci=[round(v, 4) for v in pb['ci95']], folds=per); print(nm, json.dumps(rep[nm]), flush=True)
    json.dump(rep, open(OUT + '.json', 'w'), indent=1); print('done', flush=True)


if __name__ == '__main__':
    main()
