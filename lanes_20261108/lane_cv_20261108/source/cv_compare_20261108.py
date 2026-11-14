"""CV lane (2026-11-08). Read new graph runs on CV against the best pipeline on the same rows.
NEW graph = mean of the given lane runs (graph_runs/<name>.npz), CV column only; HEX and CD stay as in the reference. Two reads:
 (a) graph level: REF_GRAPH (three shipped graph runs) vs the same with the CV column from NEW;
 (b) best level: BEST + (NEW - GRAPH) on CV, against BEST.
Rows: folds given (default 0 and 4). Paired donor bootstrap on CV APE and on the mean.
usage: cv_compare_20261108.py RUN[,RUN...] [FOLDS] [--replace] (--replace: NEW replaces GRAPH in BEST instead of adding the delta, same thing here)."""
import sys, json
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import best_pipeline_20261106 as BP
E = BP.E; y = BP.y; fold = BP.fold; donors = BP.donors; LO = BP.C.X / 'lane_cv_20261108/graph_runs'
runs = sys.argv[1].split(','); FOLDS = [int(c) for c in (sys.argv[2] if len(sys.argv) > 2 else '04')]
Z = [np.load(LO / f'{r}.npz') if not r.startswith('/') else np.load(r) for r in runs]
NEW = {f: (np.mean([z[f'f{f}'] for z in Z], 0).astype(np.float64) if all(f'f{f}' in z for z in Z) else BP.GRAPH[f].copy()) for f in range(5)}
rows = np.isin(fold, FOLDS)


def cvcol(base, new_cv):
    return {f: np.c_[base[f][:, 0], new_cv[f], base[f][:, 2]] for f in range(5)}


def read(nm, p, ref):
    a, a0 = E.ape(p, y), E.ape(ref, y); s, s0 = E.official(a[rows]), E.official(a0[rows])
    cvd = a[rows][:, 1] - a0[rows][:, 1]; ok = np.isfinite(cvd); g = np.asarray(donors)[rows][ok]; ug, inv = np.unique(g, return_inverse=True); rng = np.random.default_rng(0)
    sums = np.bincount(inv, cvd[ok]); cnt = np.bincount(inv); bs = []
    for _ in range(2000):
        k = rng.integers(len(ug), size=len(ug)); bs.append(sums[k].sum() / cnt[k].sum())
    o = dict(name=nm, CV=round(s['CV'], 3), CV_ref=round(s0['CV'], 3), dCV=round(s['CV'] - s0['CV'], 4), ci=[round(float(q), 4) for q in np.percentile(bs, [2.5, 97.5])], mean=round(s['mean'], 4), dmean=round(s['mean'] - s0['mean'], 4), per_fold={f: round(E.official(a[fold == f])['CV'] - E.official(a0[fold == f])['CV'], 3) for f in FOLDS}); print(json.dumps(o), flush=True); return o


new_cv = {f: NEW[f][:, 1] for f in range(5)}
read('graph level: CV from ' + '+'.join(runs), BP.pipe(cvcol(BP.GRAPH, new_cv)), BP.REF_GRAPH)
read('best level: BEST + (new - graph) on CV', BP.pipe(cvcol(BP.BEST, {f: BP.BEST[f][:, 1] + new_cv[f] - BP.GRAPH[f][:, 1] for f in range(5)})), BP.REF_BEST)
read('best level, half delta', BP.pipe(cvcol(BP.BEST, {f: BP.BEST[f][:, 1] + 0.5 * (new_cv[f] - BP.GRAPH[f][:, 1]) for f in range(5)})), BP.REF_BEST)
read('best level, avg old+new graph', BP.pipe(cvcol(BP.BEST, {f: BP.BEST[f][:, 1] + 0.5 * (new_cv[f] - BP.GRAPH[f][:, 1]) for f in range(5)})), BP.REF_BEST) if False else None
