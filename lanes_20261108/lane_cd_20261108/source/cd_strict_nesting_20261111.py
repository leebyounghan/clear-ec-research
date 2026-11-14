"""Strict-nesting check (coordinator 11-11): the freeze v3 CD value 6.1741 and the best pipeline's 6.2639 with every fitted constant taken from
the four folds other than the scored one. In the standard reads the stack weights, mask calibration and stretch are already fit on the other
four folds, but two inherited constants are not: P1's calibration SCALE (one global scalar fit on all five folds) and recipe A's LOFO scalars
(per row by its own fold, so for outer fold f the training rows of fold g carry a scalar fit on folds != g, which includes fold f).
Strict version: for each outer fold f, P1 and recipe A are rescaled by scalars fit (weighted median, as the originals) on folds != f only,
then the same weight grid, mask slot and stretch as the freeze search; the best pipeline's CD (hybrid B CD = 0.5 recipe A + 0.5 P1, then the
nested stretch) is rebuilt the same way."""
import os, itertools, json, numpy as np
os.environ['A2SNAP'] = '1'
import cd_read_20261108 as R
import cd_stack_20261109 as K
from cd_common import *
rows = np.isin(fold, range(5))
SCALE_CD = 0.9970446110788349
P1raw = R.P1 / SCALE_CD
Afin_raw = 0.5 * (R.seeds[46][:, 0] + R.seeds[47][:, 0])          # recipe A TTA e20 seed mean before scalars
Asnap_raw = R.A2 / R.sc                                            # recipe A snapshot before scalars
D = np.exp(0.5 * np.log(R.member('dinov3_s46_tta')) + 0.5 * np.log(R.member('dinov3_s47_tta'))); V = R.member('vitl_s4647_tt')
Bcd = R.b['B'][:, 0]
recon = 0.5 * Afin_raw * R.sc + 0.5 * R.P1
print('hybrid B CD = 0.5 A(LOFO) + 0.5 P1(SCALE)? max rel diff', float(np.nanmax(np.abs(recon / Bcd - 1))), flush=True)


def wmed_scalar(p, tr):
    ok = tr & (yc > 0); return float(C.wmed(yc[ok] / p[ok], p[ok] / yc[ok]))


def search(Lc, tr):
    grids = []
    for a in np.round(np.arange(0.25, 0.751, 0.05), 3):
        for mw in itertools.product(np.round(np.arange(0, 0.401, 0.05), 3), repeat=len(Lc) - 2):
            r = 1 - a - sum(mw)
            if r >= 0.099: grids.append((round(r, 3), a) + tuple(mw))
    best = None
    for wt in grids:
        q = np.exp(sum(w * l for w, l in zip(wt, Lc))); e = cd(apply1(q, fit_stretch1(q, tr & good)), tr)
        if best is None or e < best[0]: best = (e, wt, q)
    q = best[2]; trm = tr & K.have & (yc > 0) & good; co = np.polyfit(K.lm[trm], K.ly[trm], 1); Mc = np.polyval(co, K.lm); b2 = None
    for wmk in (0, 0.05, 0.1, 0.15, 0.2):
        qq = np.where(K.have, np.exp((1 - wmk) * np.log(q) + wmk * Mc), q); p2 = fit_stretch1(qq, tr & good); e = cd(apply1(qq, p2), tr)
        if b2 is None or e < b2[0]: b2 = (e, wmk, qq, p2)
    return apply1(b2[2], b2[3]), (best[1], b2[1])


v3_std = np.load(X / 'lane_cd_20261108/reads/freeze_cd_oof_20261111.npz')['oof_cd']
v3_strict = np.full(len(yc), np.nan); ref_strict = np.full(len(yc), np.nan); info = {}
for f in range(5):
    tr = fold != f
    sP = wmed_scalar(P1raw, tr); sA = wmed_scalar(Afin_raw, tr); sS = wmed_scalar(Asnap_raw, tr)
    pred, w = search([np.log(P1raw * sP), np.log(Asnap_raw * sS), np.log(D), np.log(V)], tr)
    v3_strict[fold == f] = pred[fold == f]
    Bf = 0.5 * Afin_raw * sA + 0.5 * P1raw * sP; prm = fit_stretch1(Bf, tr & good); ref_strict[fold == f] = apply1(Bf, prm)[fold == f]
    info[f] = dict(scale_P1=round(sP, 5), scalar_A_final=round(sA, 5), scalar_A_snapshot=round(sS, 5), weights=w)
    print('fold', f, json.dumps(info[f]), flush=True)
print('standard reads: v3', round(cd(v3_std, rows), 4), '| best pipeline', round(cd(REF, rows), 4), flush=True)
rep('v3 candidate, strict nesting', v3_strict, REF, rows=rows)
rep('v3 strict vs v3 standard', v3_strict, v3_std, rows=rows)
rep('best pipeline, strict nesting (rebuilt)', ref_strict, REF, rows=rows)
rep('v3 strict vs best pipeline strict', v3_strict, ref_strict, rows=rows)
np.savez_compressed(X / 'lane_cd_20261108/reads/strict_nesting_20261111.npz', idx=idx, v3_strict=v3_strict, ref_strict=ref_strict, v3_std=v3_std, ref_std=REF)
