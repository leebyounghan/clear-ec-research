"""Member disagreement as a per-frame MAPE shift (CD lane, 2026-11-08). One parameter, no source or ID keys.
d = SD over members of log CD (P1, A2 snapshot, DINOv3 TTA, ConvNeXt TTA); prediction after the nested blend + stretch is multiplied by exp(-k * d^2 / mean d^2)
(or exp(-k * d)); k chosen on the other folds (grid), applied to the held fold. Base: SEP blend P1/A2snap/DINOv3 TTA/ConvNeXt TTA (five folds)."""
import os, json, numpy as np
os.environ['A2SNAP'] = '1'
import cd_read_20261108 as R
from cd_common import *
D = R.member('dinov3_s46_tta'); Cx = R.member('convnext_s46_tta')
Ms = [R.P1, R.A2, D, Cx]
d = np.std(np.stack([np.log(m) for m in Ms]), 0)
print('disagreement d: median', round(float(np.median(d)), 4), 'p90', round(float(np.percentile(d, 90)), 4), flush=True)
ly = np.log(np.where(yc > 0, yc, np.nan)); ok = yc > 0
ae = np.abs(np.log(R.A2) - ly)
print('Spearman d vs |log err of A2|', round(float(np.corrcoef(np.argsort(np.argsort(d[ok])), np.argsort(np.argsort(ae[ok])))[0, 1]), 3), flush=True)
base = R.read_sep(['dinov3_s46_tta', 'convnext_s46_tta'])
# rebuild base predictions (read_sep returns only the report); recompute with fixed weights per fold from its note
note = json.loads(base['note']); out = np.full(len(yc), np.nan); Lc = [np.log(m) for m in Ms]
for f in range(5):
    wt = note[str(f)]; tr = fold != f; q = np.exp(sum(w * l for w, l in zip(wt, Lc))); out[fold == f] = apply1(q[fold == f], fit_stretch1(q, tr & good))
print('base rebuilt CD', round(cd(out), 4), flush=True)
z = d ** 2 / np.mean(d ** 2)
for form, feat in (('quad', z), ('lin', d / np.mean(d))):
    adj = np.full(len(yc), np.nan); ks = {}
    for f in range(5):
        tr = (fold != f) & good; best = None
        for k in np.round(np.arange(-0.02, 0.0801, 0.005), 3):
            e = cd(out * np.exp(-k * feat), (fold != f))
            if best is None or e < best[0]: best = (e, k)
        ks[f] = best[1]; adj[fold == f] = (out * np.exp(-best[1] * feat))[fold == f]
    rep(f'disagreement shift ({form}) on SEP base', adj, out, rows=yc > 0, note=str(ks))
