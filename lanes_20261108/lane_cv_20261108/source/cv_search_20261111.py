"""CV lane (2026-11-11, PI request): from-scratch combination search for the CV column over every component with five-fold OOF predictions.
Components are CV log corrections on top of the best pipeline (BEST), per fold model, dumped by cv_stack5_20261108.py (STACK_DUMP). Two searches,
both fully nested: for outer fold f, selection and weights use only folds != f, including the stretch, which for inner fold g is fit on folds
not in {f, g} (stricter than cv_stack5, whose inner stretch for fold g also saw fold f). The outer fold is then scored once with the outer
stretch fit on folds != f, exactly as cv_stack5 does.
  grid:    coordinate descent from all-zero weights over the whole pool, grid 0..1 step 0.125, two passes
  forward: forward selection from empty; each step adds the (component, weight) that lowers the inner CV most; after each addition one
           coordinate-descent pass over the selected weights; stops when no addition lowers it
  fixed6:  the 10.1101 components with weights refit by the same strict procedure (comparison under identical nesting)
usage: cv_search_20261111.py POOLFILE (from out/)"""
import json, sys, time
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import best_pipeline_20261106 as BP
import stretch_nested_20261103 as N
import ceiling_meta_cpu_20261103 as C
from pathlib import Path
X = Path('/workspace/clear-ec-research/experiments'); LANE = X / 'lane_cv_20261108'
y = BP.y[:, 1]; fold = BP.fold; good = BP.good; ok = y > 0; logH = np.log(BP.H[:, 1]); GRIDW = np.round(np.arange(0.125, 1.0001, 0.125), 3)
z = np.load(LANE / 'out' / sys.argv[1], allow_pickle=True); NAMES = [str(n) for n in z['names']]; K = len(NAMES)
COMP = np.stack([np.stack([z[f'c{k}_f{f}'] for f in range(5)]) for k in range(K)])   # K x 5 x N
LB = np.stack([logH + BP.BEST[f][:, 1] for f in range(5)])                          # 5 x N, base log CV of fold-model f
SG = N.GRID


def stretch_fit(lp, rows):
    """stretch_nested.fit_stretch for the CV column only: slope on the grid, MAPE-optimal scalar (weighted median)."""
    r = rows & ok; lb = lp[r]; mu = lb.mean(); yy = y[r]; best = None
    for g in SG:
        q = np.exp(g * (lb - mu) + mu); c = C.wmed(yy / q, q / yy); m = np.mean(np.abs(c * q - yy) / yy)
        if best is None or m < best[0]: best = (m, g, c, mu)
    return best[1:]


def apply_st(lp, prm):
    g, c, mu = prm; return c * np.exp(g * (lp - mu) + mu)


def pre(gm, w):
    return LB[gm] + np.tensordot(w, COMP[:, gm, :], axes=1) if w.any() else LB[gm].copy()


def inner_score(w, f):
    """mean CV APE over rows of folds != f; inner fold g scored with a stretch fit on folds not in {f, g}."""
    tot = 0.0; n = 0
    for g in range(5):
        if g == f: continue
        lp = pre(g, w); prm = stretch_fit(lp, (fold != g) & (fold != f) & good); sel = (fold == g) & ok
        p = apply_st(lp[sel], prm); tot += np.sum(np.abs(p - y[sel]) / y[sel]); n += sel.sum()
    return tot / n * 100


def score_all(w):   # deploy-style objective on all folds (for the all-fold weights): inner fold g stretch on folds != g
    tot = 0.0; n = 0
    for g in range(5):
        lp = pre(g, w); prm = stretch_fit(lp, (fold != g) & good); sel = (fold == g) & ok
        p = apply_st(lp[sel], prm); tot += np.sum(np.abs(p - y[sel]) / y[sel]); n += sel.sum()
    return tot / n * 100


def coord(w, obj, idx, passes=2):
    cur = obj(w)
    for _ in range(passes):
        for k in idx:
            best = (cur, w[k])
            for gv in np.r_[0.0, GRIDW]:
                if gv == w[k]: continue
                t = w.copy(); t[k] = gv; s = obj(t)
                if s < best[0] - 1e-9: best = (s, gv)
            w[k] = best[1]; cur = best[0]
    return w, cur


def forward(obj):
    w = np.zeros(K); cur = obj(w); sel = []; trace = []
    while True:
        best = None
        for k in range(K):
            if k in sel: continue
            for gv in GRIDW:
                t = w.copy(); t[k] = gv; s = obj(t)
                if s < cur - 1e-9 and (best is None or s < best[0]): best = (s, k, gv)
        if best is None: break
        sel.append(best[1]); w[best[1]] = best[2]; w, cur = coord(w, obj, sel, passes=1); trace.append((NAMES[best[1]], round(cur, 4)))
    return w, cur, trace


def outer(method, fixed=None):
    P = np.full(len(y), np.nan); W = {}; info = {}
    for f in range(5):
        obj = lambda w, f=f: inner_score(w, f)
        if method == 'grid': w, _ = coord(np.zeros(K), obj, list(range(K)))
        elif method == 'forward': w, _, tr = forward(obj); info[f] = tr
        else: w, _ = coord(np.zeros(K), obj, fixed)
        lp = pre(f, w); prm = stretch_fit(lp, (fold != f) & good); P[fold == f] = apply_st(lp[fold == f], prm); W[f] = w
    a = np.abs(P[ok] - y[ok]) / y[ok] * 100; ref = np.abs(BP.REF_BEST[ok, 1] - y[ok]) / y[ok] * 100
    perf = {f: round(float(np.mean(np.abs(P[(fold == f) & ok] - y[(fold == f) & ok]) / y[(fold == f) & ok]) * 100 - np.mean(np.abs(BP.REF_BEST[(fold == f) & ok, 1] - y[(fold == f) & ok]) / y[(fold == f) & ok]) * 100), 4) for f in range(5)}
    return float(a.mean()), float(a.mean() - ref.mean()), perf, W, info, P


if __name__ == '__main__':
    t0 = time.time(); print('pool', K, NAMES, flush=True)
    six = [NAMES.index(n) for n in NAMES if n in ('dino', 'fz:spectrum', 'fz:cov_sqrt', 'fz:autocorr') or n.startswith('graph:tta_s46_tta4_5f+') or n.startswith('graph:gord_s46_tta4_5f+')]
    out = {}
    for m in ('fixed6', 'forward', 'grid'):
        cv, d, perf, W, info, P = outer(m, six)
        picks = {f: {NAMES[k]: float(W[f][k]) for k in range(K) if W[f][k] > 0} for f in range(5)}
        cnt = {}
        for f in range(5):
            for n in picks[f]: cnt[n] = cnt.get(n, 0) + 1
        out[m] = dict(cv=round(cv, 4), delta_vs_best=round(d, 4), per_fold=perf, picks=picks, pick_count=dict(sorted(cnt.items(), key=lambda kv: -kv[1])), forward_trace={f: info.get(f) for f in info})
        print(json.dumps({m: out[m]}), flush=True); np.savez_compressed(LANE / f'out/search_{m}_20261111.npz', P=P, names=np.array(NAMES), **{f'w{f}': W[f] for f in range(5)})
    json.dump(out, open(LANE / 'out/search_20261111.json', 'w'), indent=1); print('done', round(time.time() - t0), 's', flush=True)
