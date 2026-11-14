"""Leave-one-source-out shift split (backbone lane, 2026-11-10; coordinator / PI). Source = ID format, used ONLY to define the
evaluation split, never as a model input (con-no-source-id-shortcuts). Sources: P1 = 'NNNN-YY' IDs, P2 = 'NNNN-NNNN[ -]aaaa' IDs.
For each source, a donor-grouped ~20 % hold-out (seed 1110). Split A trains on P1 minus its hold-out, split B on P2 minus its hold-out;
'validation' = that source's hold-out + every row of the other source (both scored, separated by the read).
Writes data/source_split_20261110.npz (membership) and data/split_src{A,B}_20261110.npz (train / validation for the runners)."""
import re, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/eval_harness_20261031/source')
import eval_oof_20261031 as E
R = Path('/workspace/clear-ec-research'); OUT = R / 'experiments/lane_backbone_20261108/data'
with np.load(R / 'datasets/clear_ec_official/cache_native_meta_512.npz', allow_pickle=True) as z:
    ids, Y = z['ids'].astype(str), z['Y'].astype(np.float64)
N = len(ids); norm = [re.sub(r'\s+', ' ', i.strip()) for i in ids]
p1 = np.array([bool(re.fullmatch(r'\d{4}-\d{2}', i)) for i in norm]); p2 = np.array([bool(re.fullmatch(r'\d{4}-\d{4}[ -][0-9A-Za-z]{4}', i)) for i in norm])
assert not (p1 & p2).any(); print(json.dumps(dict(P1=int(p1.sum()), P2=int(p2.sum()), neither=int((~p1 & ~p2).sum()), neither_ids=[norm[i] for i in np.where(~p1 & ~p2)[0][:5]])))
donor = E.donors(np.array(norm))  # slide_id groups (folds are donor-grouped)
pos = np.isfinite(Y).all(1) & (Y > 0).all(1)
rng = np.random.default_rng(1110); hold = np.zeros(N, bool)
for src in (p1, p2):
    d = np.unique(donor[src]); rng.shuffle(d); target = 0.2 * src.sum(); n = 0
    for g in d:
        if n >= target: break
        m = src & (donor == g); hold |= m; n += m.sum()
# donors must not straddle sources' hold-out and train within a source
for src in (p1, p2):
    assert not np.intersect1d(donor[src & hold], donor[src & ~hold]).size
b = E.load_base(); pos_b = {int(i): k for k, i in enumerate(b['idx'])}; rows = np.array([pos_b.get(i, -1) for i in range(N)])
for name, src in (('P1', p1), ('P2', p2)):
    m = src & (rows >= 0); e = E.official(E.ape(b['B'][rows[m]], b['y'][rows[m]]))
    print(json.dumps(dict(source=name, rows=int(src.sum()), holdout=int((src & hold).sum()), donors=int(len(np.unique(donor[src]))), hybridB_oof={k: round(e[k], 3) for k in ('CD', 'CV', 'HEX', 'mean')})))
idx = np.arange(N)
np.savez_compressed(OUT / 'source_split_20261110.npz', P1=idx[p1], P2=idx[p2], holdout=idx[hold], donor=donor, ids=np.array(norm))
for tag, src, oth in (('A', p1, p2), ('B', p2, p1)):
    tr = idx[src & ~hold & pos]; va = np.sort(np.r_[idx[src & hold], idx[oth]])
    assert not np.intersect1d(tr, va).size and not np.intersect1d(donor[tr], donor[va]).size
    np.savez_compressed(OUT / f'split_src{tag}_20261110.npz', train=tr, validation=va, primary_validation=va)
    print(json.dumps(dict(split=tag, train=len(tr), validation=len(va), val_in_source=int((src & hold).sum()), val_cross=int(oth.sum()))))
