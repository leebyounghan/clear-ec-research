"""Cell-route lane for the backbone lane (2026-11-09): size of the second-order teacher leak, measured on a proxy learner where both teachers exist.
Learner: histogram gradient boosting (memorizing setting: L1 on log label, lr 0.1, 300 iterations, 31 leaves, min leaf 20) on 148 label-free columns
(cell-route point statistics + 4 whole-image statistics). For each outer fold f and metric: clean teacher for fold-g rows = model fit on folds not in
{f, g}; leaky teacher = the usual OOF model fit on folds != g (it saw fold f). Student = same learner fit on folds != f to exp(0.5 log y + 0.5 log t)
(bb_runner's --teacher-a 0.5 mix), scored on fold f. Leak = student MAPE with the leaky teacher minus with the clean one (negative = optimistic).
Option B analogue (insample): teacher = the fold-f model on its own training rows. Also the learner's memorization gap (training-row vs unseen-row MAPE), to compare with recipe A's 0.52."""
import sys, json
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import best_pipeline_20261106 as BP
L = '/workspace/clear-ec-research/experiments/lane_cellroute_20261109/feats/'; idx = BP.idx.astype(int); y, fold, good = BP.y, BP.fold, BP.good
X = np.column_stack([np.load(L + f + '.npz')['F'][idx] for f in ('cellroute_feats', 'trim_feats', 'click_feats_P2_P5_m1_K8')])
z = np.load('/workspace/clear-ec-research/experiments/eval_harness_20261031/image_stats.npz'); pos = {int(i): k for k, i in enumerate(z['idx'])}
X = np.c_[X, np.column_stack([z[c][[pos[int(i)] for i in idx]] for c in ('mean', 'contrast', 'black_frac', 'focus')])]
mk = lambda: HistGradientBoostingRegressor(loss='absolute_error', learning_rate=0.1, max_iter=300, max_leaf_nodes=31, min_samples_leaf=20, early_stopping=False, random_state=0)
mape = lambda p, t: float(np.mean(np.abs(p / t - 1)) * 100)
out = {}
for j, nm in enumerate(('CD', 'CV', 'HEX')):
    ok = good & (y[:, j] > 0); ly = np.log(np.where(y[:, j] > 0, y[:, j], 1.0))
    oof = np.full(len(y), np.nan); gaps = []
    for g in range(5):
        tr = ok & (fold != g); m = mk().fit(X[tr], ly[tr]); p = m.predict(X); oof[fold == g] = p[fold == g]
        te = ok & (fold == g); gaps.append(mape(np.exp(p[te]), y[te, j]) - mape(np.exp(p[tr]), y[tr, j]))
    res = []
    for f in range(5):
        clean = np.full(len(y), np.nan)
        for g in [k for k in range(5) if k != f]:
            tr = ok & (fold != f) & (fold != g); clean[fold == g] = mk().fit(X[tr], ly[tr]).predict(X[fold == g])
        tr = ok & (fold != f); te = ok & (fold == f); r = {}
        insamp = mk().fit(X[tr], ly[tr]).predict(X)  # option B analogue: the fold-f model on its own training rows (saw them, never saw fold f)
        for tn, t in (('labels', ly), ('clean', 0.5 * ly + 0.5 * clean), ('leaky', 0.5 * ly + 0.5 * oof), ('insample', 0.5 * ly + 0.5 * insamp)):
            r[tn] = mape(np.exp(mk().fit(X[tr], t[tr]).predict(X[te])), y[te, j])
        res.append(r)
    lk = [r['leaky'] - r['clean'] for r in res]
    out[nm] = dict(memorization_gap=round(float(np.mean(gaps)), 3), student_labels=round(float(np.mean([r['labels'] for r in res])), 3), student_clean=round(float(np.mean([r['clean'] for r in res])), 3),
                   student_leaky=round(float(np.mean([r['leaky'] for r in res])), 3), student_insample=round(float(np.mean([r['insample'] for r in res])), 3), leak_leaky_minus_clean=round(float(np.mean(lk)), 4), leak_per_fold=[round(v, 4) for v in lk])
    print(nm, json.dumps(out[nm]), flush=True)
json.dump(out, open('/workspace/clear-ec-research/experiments/lane_cellroute_20261109/teachers/leak_proxy_v2.json', 'w'), indent=1)
