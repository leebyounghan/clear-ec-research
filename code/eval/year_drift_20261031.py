"""Does the hybrid's gain over Phase-1 depend on acquisition year? (2026-10-31, CPU, OOF only)"""
import json, sys
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/eval_harness_20261031/source')
import eval_oof_20261031 as E

b = E.load_base()
y = b['y']
eA, eP, eB = (E.ape(b[k], y) for k in ('A', 'P1', 'B'))
year = np.array([E.year_of(i) for i in b['ids']])
g = E.donors(b['ids'])
out = {'per_year': {}, 'groups': {}}
for yr in np.unique(year):
    r = year == yr
    out['per_year'][int(yr)] = dict(n=int(r.sum()), **{k: {m: round(v, 3) for m, v in E.official(e, r).items() if m != 'n'} for k, e in (('A', eA), ('P1', eP), ('B', eB))})
for name, r in (('2018_2022', year <= 2022), ('2023_2024', year >= 2023)):
    gg = np.unique(g[r], return_inverse=True)[1]
    out['groups'][name] = dict(n=int(r.sum()),
                               B_minus_P1=E.paired_bootstrap(eB[r], eP[r], gg), A_minus_P1=E.paired_bootstrap(eA[r], eP[r], gg))
# interaction: is the recent-year gain different from the earlier gain?
rng = np.random.default_rng(1)
G = g.max() + 1
d_img = np.nanmean(eB - eP, 1)
early, late = year <= 2022, year >= 2023
diffs = []
for _ in range(2000):
    w = rng.multinomial(G, np.full(G, 1 / G)).astype(float)[g]
    diffs.append(np.average(d_img[late], weights=w[late]) - np.average(d_img[early], weights=w[early]))
out['interaction_late_minus_early'] = dict(point=float(d_img[late].mean() - d_img[early].mean()),
                                            ci95=[float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))])
q = np.quantile(np.abs(np.log(b['A'] / b['P1'])).mean(1), 0.75)
hi = np.abs(np.log(b['A'] / b['P1'])).mean(1) >= q
out['high_disagreement_q4'] = {k: round(E.official(e, hi)['mean'], 3) for k, e in (('A', eA), ('P1', eP), ('B', eB))}
out['year_by_fold'] = {int(f): {int(yr): int(((b['fold'] == f) & (year == yr)).sum()) for yr in np.unique(year)} for f in range(5)}
json.dump(out, open(sys.argv[1], 'w'), indent=1)
print(json.dumps(out, indent=1))
