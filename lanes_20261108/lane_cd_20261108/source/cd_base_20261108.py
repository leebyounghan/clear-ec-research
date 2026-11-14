"""CPU baseline for the CD lane: what the stored members give on CD alone, nested."""
import json, numpy as np
from cd_common import *
b = E.load_base(); seeds, single, hist = C.load_members(b)
sc = np.array([json.load(open(X / 'recipeA_analysis_20261028/hybrid_oof.json'))['recipeA_lofo_scalars'][str(f)][0] for f in fold])
print('seeds', sorted(seeds), 'hist', sorted(hist), flush=True)
A2 = 0.5 * (seeds[46][:, 0] + seeds[47][:, 0]) * sc
A3 = np.mean([seeds[s][:, 0] for s in sorted(seeds)], 0) * sc
P1 = b['P1'][:, 0]; Bc = b['B'][:, 0]
print('check A2 vs base A', np.nanmax(np.abs(A2 / b['A'][:, 0] - 1)), flush=True)
ALL = (yc > 0)
for nm, p in (('ref_best', REF), ('B_raw', Bc), ('A2', A2), ('A3', A3), ('P1', P1)):
    print(nm, 'all', round(cd(p), 4), 'screen', round(cd(p, SCREEN), 4), 'old', round(cd(p, ~SRC_NEW), 4), 'new', round(cd(p, SRC_NEW), 4), flush=True)
rep('B stretched (should be ~0)', stretch(Bc), rows=ALL)
# nested A-weight
def wnest(Am, Pm, geo=False):
    out = np.full_like(Am, np.nan); ws = {}
    for f in range(5):
        best = None
        for w in np.round(np.arange(0.3, 1.001, 0.05), 2):
            q = np.exp(w * np.log(Am) + (1 - w) * np.log(Pm)) if geo else w * Am + (1 - w) * Pm
            s = stretch(q)   # nested stretch inside; score on folds != f only
            m = cd(s, fold != f)
            if best is None or m < best[0]: best = (m, w)
        ws[f] = best[1]; w = best[1]
        q = np.exp(w * np.log(Am) + (1 - w) * np.log(Pm)) if geo else w * Am + (1 - w) * Pm
        out[fold == f] = stretch(q)[fold == f]
    return out, ws
for nm, Am in (('A2', A2),):
    for geo in (False, True):
        p, ws = wnest(Am, P1, geo); rep(f'{nm} weight nested geo={geo}', p, rows=ALL, note=str(ws)); rep(f'{nm} weight nested geo={geo} screen', p)
rep('B src stretch', stretch_src(Bc), rows=ALL)
rep('A3+P1 0.5 stretch', stretch(0.5 * A3 + 0.5 * P1), rows=ALL)
for n, h in hist.items():
    print('hist', n, round(cd(h[:, 0]), 4), flush=True)
