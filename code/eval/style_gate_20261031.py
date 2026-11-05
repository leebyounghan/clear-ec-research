"""Style-gated hybrid weight (2026-10-31). Pre-registered before running:
  stratum   : black-frame style = fraction of near-black pixels (< 5 on a 2x-downsampled frame) > 0.01, else full-frame
  model     : out = w_s * A + (1 - w_s) * P1, one w per stratum, grid 0..1 step 0.05, fitted by the official score
              on the training folds only (leave-one-fold-out), applied to the held fold
  adopt if  : dev folds 0-3 mean delta vs B <= -0.02 AND fold-4 delta <= 0
  caveat    : the stratifier was chosen after seeing year slices that included fold 4, so fold 4 is not pristine
"""
import json, sys
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/eval_harness_20261031/source')
import eval_oof_20261031 as E

b = E.load_base(); y = b['y']; st = E.image_stats(b)
style = st['black_frac'] > 0.01
grid = np.round(np.arange(0, 1.0001, 0.05), 2)

def score(p, rows):
    return E.official(E.ape(p, y), rows)['mean']

pred = np.zeros_like(b['B']); ws = {}
for f in range(5):
    tr, te = b['fold'] != f, b['fold'] == f
    ws[f] = {}
    for s in (False, True):
        r_tr = tr & (style == s)
        w = grid[np.argmin([score(g * b['A'] + (1 - g) * b['P1'], r_tr) for g in grid])]
        ws[f][str(s)] = float(w)
        r_te = te & (style == s)
        pred[r_te] = w * b['A'][r_te] + (1 - w) * b['P1'][r_te]
eG, eB = E.ape(pred, y), E.ape(b['B'], y)
g = E.donors(b['ids'])
def paired(rows):
    return E.paired_bootstrap(eG[rows], eB[rows], np.unique(g[rows], return_inverse=True)[1])
dev, f4 = b['fold'] <= 3, b['fold'] == 4
res = dict(style_fraction=float(style.mean()), w_by_held_fold=ws,
           dev_folds_0_3=paired(dev), fold4=paired(f4),
           per_fold_delta={f: round(score(pred, b['fold'] == f) - score(b['B'], b['fold'] == f), 4) for f in range(5)},
           by_style={str(s): paired(style == s) for s in (False, True)},
           year_2023_2024=paired(np.array([E.year_of(i) >= 2023 for i in b['ids']])))
res['adopt'] = bool(res['dev_folds_0_3']['delta'] <= -0.02 and res['fold4']['delta'] <= 0)
np.savez(sys.argv[1].replace('.json', '.npz'), indices=b['idx'], pred=pred)
json.dump(res, open(sys.argv[1], 'w'), indent=1)
print(json.dumps(res, indent=1))
