"""Row weights for the hard-image probe (coordinator 13:27 EST 11-14). Disagreement = SD over the four tier-p CD members' out-of-fold log
predictions (P1_in, A_in, D, V of reads/cd_tier_t1_20261112.npz: strict five-fold inputs, no label used). For fold f, the top decile of
the fold's TRAINING rows (fold != f) gets weight 2, every other row 1 (validation rows also 1; the runner ignores them in the loss).
Writes data/rowweight_hard2_f<f>_20261114.npz {indices = memmap rows, Z, sd} for bb_runner --row-weight (a LOSS weight, normalised by
the mean training weight)."""
import numpy as np, json
L = 'experiments/lane_cd_20261108/'
z = np.load(L + 'reads/cd_tier_t1_20261112.npz'); ix = z['idx'].astype(np.int64); fold = z['fold']
Lg = np.log(np.stack([z['P1_in'], z['A_in'], z['D'], z['V']], 1)); sd = Lg.std(1, ddof=1); assert np.isfinite(sd).all()
out = {}
for f in (0, 4):
    tr = fold != f; thr = np.quantile(sd[tr], 0.9); w = np.ones(len(sd)); w[tr & (sd >= thr)] = 2.0
    np.savez_compressed(L + f'data/rowweight_hard2_f{f}_20261114.npz', indices=ix, Z=w.astype(np.float32), sd=sd.astype(np.float32))
    out[f] = dict(rows=int(len(sd)), train_rows=int(tr.sum()), weighted_rows=int((tr & (w == 2)).sum()), sd_threshold=round(float(thr), 5),
                  sd_median_train=round(float(np.median(sd[tr])), 5), normalised_weights=[round(2 / w[tr].mean(), 4), round(1 / w[tr].mean(), 4)],
                  val_rows_top_decile_by_same_threshold=int(((fold == f) & (sd >= thr)).sum()))
print(json.dumps(out, indent=1))
