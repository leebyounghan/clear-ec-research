"""CD lane shared helpers (2026-11-08). CD only. Reference = best pipeline CD (hybrid B CD x nested log-affine stretch).
Screen rows = folds 0 and 4. Everything nested: any weight/scalar used on fold f is fit on rows of folds != f."""
import sys, re, json, warnings
import numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import best_pipeline_20261106 as BP
import stretch_nested_20261103 as N
C = BP.C; E = BP.E
y = BP.y; fold = BP.fold; idx = BP.idx; ids = BP.ids; donors = BP.donors; good = BP.good
X = C.X
LANE = X / 'lane_cd_20261108'
yc = y[:, 0]
REF = BP.REF_BEST[:, 0].copy()
SRC_NEW = np.array([bool(re.match(r'^\d{4}-\d{4}', i.strip())) for i in ids])   # YYYY-NNNN new format (source B)
SCREEN = (fold == 0) | (fold == 4)
GRID = N.GRID

def ape(p): return np.where(yc > 0, np.abs(p - yc) / np.where(yc > 0, yc, 1), np.nan) * 100
def cd(p, rows=None):
    e = ape(p); ok = yc > 0 if rows is None else (rows & (yc > 0)); return float(np.nanmean(e[ok]))
def fit_stretch1(p, rows):
    ok = rows & (yc > 0); lb = np.log(p[ok]); mu = lb.mean(); best = None
    for g in GRID:
        q = np.exp(g * (lb - mu) + mu); c = C.wmed(yc[ok] / q, q / yc[ok]); m = np.mean(np.abs(c * q - yc[ok]) / yc[ok])
        if best is None or m < best[0]: best = (m, g, c, mu)
    return best[1:]
def apply1(p, prm):
    g, c, mu = prm; return c * np.exp(g * (np.log(p) - mu) + mu)
def stretch(p, extra_rows=None):
    """nested per-fold stretch of a CD vector (fit on folds != f & good)"""
    out = np.full_like(p, np.nan)
    for f in range(5):
        tr = (fold != f) & good
        if extra_rows is not None: tr = tr & extra_rows
        out[fold == f] = apply1(p[fold == f], fit_stretch1(p, tr))
    return out
def stretch_src(p):
    out = np.full_like(p, np.nan)
    for f in range(5):
        for s in (False, True):
            m = SRC_NEW == s; out[(fold == f) & m] = apply1(p[(fold == f) & m], fit_stretch1(p, (fold != f) & good & m))
    return out
def boot(p, ref, rows=None, reps=2000):
    rows = (yc > 0) if rows is None else rows & (yc > 0)
    e1, e0 = ape(p)[rows], ape(ref)[rows]; g = donors[rows]
    u, inv = np.unique(g, return_inverse=True); d = e1 - e0
    s = np.bincount(inv, d, len(u)); n = np.bincount(inv, None, len(u)); rng = np.random.default_rng(0)
    bs = []
    for _ in range(reps):
        k = rng.integers(0, len(u), len(u)); bs.append(s[k].sum() / n[k].sum())
    return float(d.mean()), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
def rep(name, p, ref=REF, rows=None, note=''):
    rows_ = SCREEN if rows is None else rows
    d, ci = boot(p, ref, rows_)
    o = dict(name=name, cd=round(cd(p, rows_), 4), ref=round(cd(ref, rows_), 4), delta=round(d, 4), ci=[round(v, 4) for v in ci],
             folds={f: round(cd(p, fold == f) - cd(ref, fold == f), 3) for f in range(5) if (rows_ & (fold == f)).any()},
             old=round(cd(p, rows_ & ~SRC_NEW) - cd(ref, rows_ & ~SRC_NEW), 3), new=round(cd(p, rows_ & SRC_NEW) - cd(ref, rows_ & SRC_NEW), 3), note=note)
    print(json.dumps(o), flush=True); return o
