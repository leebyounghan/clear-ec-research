"""Per-epoch CD value of the family members (folds 0, 4): alone (cross-fit stretch) and added to R2 (P1/A2 geometric reweight) with a cross-fit weight.
Also 3-epoch windows. Guides the schedule for folds 1-3; the choice is made on these rows, so it is flagged as optimistic for folds 0 and 4."""
import json, numpy as np
from cd_common import *
import cd_pool_screen2_20261108 as P2
P1, A2, R2 = P2.P1, P2.A2, P2.R2
g3 = [(round(1 - a - c, 2), round(a, 2), round(c, 2)) for a in np.arange(0.3, 0.901, 0.05) for c in np.arange(0, 0.501, 0.05) if a + c <= 1.0001]
out = {}
for k, (d0, d4, arch) in P2.FAM.items():
    rows = []
    for e in range(1, 21):
        M = P2.fam(d0, d4, arch, [e]); p, w = P2.xblend([P1, A2, M], g3); a = P2.xs(M)
        rows.append((e, round(cd(a, SCREEN), 3), round(cd(p, SCREEN) - cd(R2, SCREEN), 4), round(cd(p, fold == 0) - cd(R2, fold == 0), 3), round(cd(p, fold == 4) - cd(R2, fold == 4), 3)))
    for lo in (4, 6, 8, 10, 12, 14, 16):
        M = P2.fam(d0, d4, arch, range(lo, lo + 5)); p, w = P2.xblend([P1, A2, M], g3)
        rows.append((f'{lo}-{lo+4}', round(cd(P2.xs(M), SCREEN), 3), round(cd(p, SCREEN) - cd(R2, SCREEN), 4), round(cd(p, fold == 0) - cd(R2, fold == 0), 3), round(cd(p, fold == 4) - cd(R2, fold == 4), 3)))
    out[k] = rows
    print(k, flush=True)
    for r in rows: print('  epoch', r[0], 'alone', r[1], 'added', r[2], 'f0', r[3], 'f4', r[4], flush=True)
json.dump(out, open(LANE / 'reads/cd_epoch_curve.json', 'w'), indent=1)
