#!/usr/bin/env python3
"""D-CD-PRETRAIN-SCREEN (PI 22:42 EST 11-13): DINOv3-B/16 LeJEPA SSL, fold FOLD (argv 4; 99 = the 8952-row full-data split), seed 1003, 5 epochs, exactly family_cd_20261108.run_ssl
(imported, not edited) with one change patched in memory for this process only:
  S1: photometric views gain uniform 0.8-1.25 and offset +-0.08 (was 0.9-1.1 and +-0.03; flips unchanged)
  S2: SIGReg weight lamb 0.05 (was 0.02) in loss = (1 - lamb) * invariance + lamb * sigreg
The SSL script has no crop or mask view. usage: ssl_variant_fold_20261113.py S1|S2 OUTDIR WORKERS FOLD"""
import sys, argparse
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, 'experiments/lane_cd_20261108/source')
import family_cd_20261108 as Fm
VAR, out, workers, FOLD = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
if VAR == 'S1':
    def augmented_view(a, seed, epoch, index, view):
        rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, index, view])); b = a
        if rng.random() < .5: b = b[:, ::-1]
        if rng.random() < .5: b = b[::-1, :]
        b = np.clip(b * rng.uniform(.8, 1.25) + rng.uniform(-.08, .08), 0, 1)
        return torch.from_numpy(np.ascontiguousarray(b[None]))
    Fm.S.augmented_view = augmented_view
elif VAR == 'S2':
    Fm.pilot.RECIPE['lamb'] = 0.05
else:
    raise SystemExit('VAR must be S1 or S2')
print({'variant': VAR, 'lamb': Fm.pilot.RECIPE['lamb'], 'view_fn': Fm.S.augmented_view.__module__}, flush=True)
a = argparse.Namespace(family='dinov3_b16', fold=FOLD, micro=0, ssl_epochs=5, workers=workers)
print('ssl done', Fm.run_ssl(a, out), flush=True)
