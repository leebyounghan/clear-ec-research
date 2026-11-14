"""Fallback builder of the swapped recipe A for D-SYN-003 (HEX lane, 2026-11-11), used only if the CD lane's file
(A-CD-SWAPA-PLAN) does not appear: the CD lane's spec, A = 0.5 (TTA4 e20 s46 + s47) x the row's fold LOFO scalars
(recipeA_analysis_20261028/hybrid_oof.json), with the synthetic-pretrained s46 final four-flip TTA (runs/synthft_ra_f{0,4}_s46) on
folds 0 and 4 and plain elsewhere. Checks that the plain reconstruction equals hybrid_oof A on every row first.
usage: build_swapA_self.py OUT.npz"""
import sys, json
import numpy as np
X = '/workspace/clear-ec-research/experiments'
z = np.load(f'{X}/recipeA_analysis_20261028/hybrid_oof.npz'); idx = z['indices']; fold = z['fold']; A_plain = z['A_recipeA_tta_scalar'].astype(float)
pos = {int(i): k for k, i in enumerate(idx)}
sc = {int(k): np.array(v) for k, v in json.load(open(f'{X}/recipeA_analysis_20261028/hybrid_oof.json'))['recipeA_lofo_scalars'].items()}
RECIPE = {46: {0: 'phase2_pooling_replication_20261026/f0_s46_std', 1: 'oof_recipeA_20261027/fold1/regression_std_s46', 2: 'oof_recipeA_20261027/fold2/regression_std_s46_v2',
               3: 'oof_recipeA_20261027/fold3/regression_std_s46', 4: 'phase2_pooling_20261020/arms/std'},
          47: {0: 'phase2_pooling_replication_20261026/f0_s47_std', 4: 'phase2_pooling_replication_20261026/f4_s47_std', **{f: f'oof_recipeA_20261027/fold{f}/regression_std_s47' for f in (1, 2, 3)}}}


def member(seed):
    arr = np.full((len(idx), 3), np.nan)
    for f, root in RECIPE[seed].items():
        t = np.load(f'{X}/{root}/runs/fold{f}/cpdino_vitb/predictions_tta4.npz'); arr[[pos[int(i)] for i in t['indices']]] = t['pred_tta4']
    return arr


m46, m47 = member(46), member(47); scal = np.stack([sc[int(f)] for f in fold])
rec = 0.5 * (m46 + m47) * scal; d = float(np.max(np.abs(rec - A_plain) / A_plain)); print('plain reconstruction max rel diff', d, flush=True); assert d < 1e-6
syn = m46.copy(); swapped = np.zeros(len(idx), bool)
for f in (0, 4):
    t = np.load(f'{X}/lane_backbone_20261108/runs/synthft_ra_f{f}_s46/raw/runs/fold{f}/cpdino_b8/predictions_final_tta.npz'); key = 'pred_tta4' if 'pred_tta4' in t.files else 'pred'
    r = np.array([pos[int(i)] for i in t['indices']]); assert (fold[r] == f).all(); syn[r] = t[key]; swapped[r] = True
A = 0.5 * (syn + m47) * scal
np.savez_compressed(sys.argv[1], indices=idx, fold=fold, A=A, A_plain=A_plain, swapped=swapped)
print('written', sys.argv[1], 'swapped rows', int(swapped.sum()), 'mean log ratio HEX on swapped rows', float(np.mean(np.log(A[swapped, 2] / A_plain[swapped, 2]))), flush=True)
