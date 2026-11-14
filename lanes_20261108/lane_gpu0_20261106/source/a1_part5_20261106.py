"""ID spelling inside source B: is the hyphen form a separate batch? (lane GPU 0, 2026-11-06)"""
import re, sys
import numpy as np
sys.path.insert(0, 'experiments/meta_ceiling_20261103/source')
import best_pipeline_20261106 as BP
E = BP.E
ids, y, P = BP.ids, BP.y, BP.REF_BEST
err = E.ape(P, y); res = np.log(P) - np.log(np.where(y > 0, y, np.nan))
sa = np.array([bool(re.fullmatch(r'\d+-\d\d', i)) for i in ids])
form = np.array(['A' if a else ('hyphen' if re.match(r'\d{4}-\d+-', i) else ('space' if re.match(r'\d{4}-\d+ ', i) else 'other')) for i, a in zip(ids, sa)])
yr = np.array([0 if a else int(i[:4]) for i, a in zip(ids, sa)])
ser = np.array([0 if a else int(re.match(r'\d{4}-(\d+)', i).group(1)) for i, a in zip(ids, sa)])
print('forms', dict(zip(*[x.tolist() for x in np.unique(form, return_counts=True)])), 'others', ids[form == 'other'][:5].tolist())
for v in sorted(set(yr[~sa])):
    for f in ('hyphen', 'space', 'other'):
        m = (yr == v) & (form == f)
        if m.sum():
            e = np.nanmean(err[m], 0); b = np.nanmean(res[m], 0); med = np.nanmedian(err[m], 0); sd = np.nanstd(res[m], 0); ml = np.nanmean(np.where(y[m] > 0, y[m], np.nan), 0)
            print(v, f.ljust(7), f'n={m.sum():5d} serial {ser[m].min()}-{ser[m].max()} err {e[0]:5.2f} {e[1]:5.2f} {e[2]:5.2f} median {med[0]:4.1f} {med[1]:4.1f} {med[2]:4.1f} bias {b[0]:+.3f} {b[1]:+.3f} {b[2]:+.3f} resSD {sd[0]:.3f} {sd[1]:.3f} {sd[2]:.3f} label {ml[0]:5.0f} {ml[1]:.3f} {ml[2]:.3f}')
