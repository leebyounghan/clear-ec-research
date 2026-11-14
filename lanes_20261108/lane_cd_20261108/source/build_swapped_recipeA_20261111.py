"""Q-HEX-001 / D-SYN-003: recipe A with the synthetic-pretrained s46 in place of plain s46, as a file for the HEX lane (and the CD read).
Construction = recipeA_analysis_20261028/hybrid_oof.npz A_recipeA_tta_scalar: per row 0.5 * (TTA4@e20 s46 + TTA4@e20 s47) x the row's fold LOFO
scalars (3 columns: CD, CV, HEX), same row order as hybrid_oof.npz ('indices'). On the folds where the synthetic run exists, s46's TTA4@e20 is
replaced by the synthetic run's predictions_final_tta.npz (e20 four-flip TTA); other rows keep the plain value. Also writes the CD lane's snapshot
form (per seed exp(0.5 log TTA4@e20 + 0.5 mean log single e8-19), same seeds and scalars) as A_snap.
usage: build_swapped_recipeA_20261111.py [SYN_ROOT_TEMPLATE] [OUT]   default root lane_backbone_20261108/runs/synthft_ra_f{f}_s46/raw, family cpdino_b8"""
import os, sys, json, numpy as np
os.environ['A2SNAP'] = '1'
import cd_read_20261108 as R
from cd_common import *
TMPL = sys.argv[1] if len(sys.argv) > 1 else 'lane_backbone_20261108/runs/synthft_ra_f{f}_s46/raw'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'lane_cd_20261108/reads/recipeA_swapped_syn46_20261111.npz'
FAM = os.environ.get('SYN_FAM', 'cpdino_b8')
H = np.load(X / 'recipeA_analysis_20261028/hybrid_oof.npz', allow_pickle=True)
hpos = [R.pos[int(i)] for i in H['indices']]          # hybrid row k -> our row
lofo = json.load(open(X / 'recipeA_analysis_20261028/hybrid_oof.json'))['recipeA_lofo_scalars']
SC = np.array([lofo[str(f)] for f in fold])          # our order, 3 columns
plain = 0.5 * (R.seeds[46] + R.seeds[47]) * SC
assert np.allclose(plain[hpos], H['A_recipeA_tta_scalar'], rtol=1e-12), 'plain reconstruction differs from hybrid_oof'


def vec3(p):
    z = np.load(p); v = np.full((len(yc), 3), np.nan); v[[R.pos[int(i)] for i in z['indices']]] = z['pred'][:, :3]; return v


t46 = R.seeds[46].copy(); r46 = np.stack([R._rec_col(46, j) for j in range(3)], 1) if hasattr(R, '_rec_col') else None
swapped = np.zeros(len(yc), bool); notes = {}
syn_tta = np.full((len(yc), 3), np.nan); syn_single = np.full((len(yc), 3), np.nan)
for f in range(5):
    d = X / TMPL.format(f=f) / f'runs/fold{f}' / FAM
    if not (d / 'predictions_final_tta.npz').exists(): continue
    p = vec3(d / 'predictions_final_tta.npz'); rows = fold == f
    singles = [d / f'predictions_epoch_{e:02d}.npz' for e in range(8, 20)]
    if all(s.exists() for s in singles):
        syn_single[rows] = np.exp(np.mean([np.log(vec3(s)) for s in singles], 0))[rows]
    syn_tta[rows] = p[rows]; swapped |= rows; notes[f] = f'final TTA + {sum(s.exists() for s in singles)} single epochs'
t46[swapped] = syn_tta[swapped]
A = 0.5 * (t46 + R.seeds[47]) * SC
# CD snapshot form (CD column only uses the CD lane's _rec; HEX/CV columns given in TTA form above)
s46_single_cd = R._rec(46).copy(); s46_single_cd[swapped] = syn_single[swapped, 0]
A_snap_cd = 0.5 * (np.exp(0.5 * np.log(t46[:, 0]) + 0.5 * np.log(s46_single_cd)) + np.exp(0.5 * np.log(R.seeds[47][:, 0]) + 0.5 * np.log(R._rec(47)))) * R.sc
np.savez_compressed(X / OUT, indices=H['indices'], ids=H['ids'], fold=H['fold'], A=A[hpos], A_plain=plain[hpos], swapped=swapped[hpos], A_snap_cd=A_snap_cd[hpos],
                    note=np.array(f'synthetic s46 from {TMPL} ({FAM}) on folds {sorted(notes)}; rows in hybrid_oof.npz order; A = 0.5*(TTA4 e20 s46 + s47) x LOFO scalars'))
print(json.dumps(dict(out=OUT, folds=notes, swapped_rows=int(swapped.sum()), max_rel_change_on_swapped=float(np.nanmax(np.abs(A[swapped] / plain[swapped] - 1))) if swapped.any() else 0.0,
                      unswapped_equal_plain=bool(np.allclose(A[~swapped], plain[~swapped])))), flush=True)
