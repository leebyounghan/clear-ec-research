import re
p = '/workspace/clear-ec-research/experiments/lane_cv_20261108/REPORT.md'; s = open(p).read()
lines = s.split('\n')
for i, l in enumerate(lines):
    if l.startswith('- Lane best (five folds, nested weights)'):
        lines[i] = ('- Lane best, one recipe on all five folds (nested weights): CV 10.114 vs best pipeline 10.151 (-0.037; folds -0.096/+0.058/-0.069/-0.055/-0.023). '
                    'Components: DINOv3 B/16 member (mean of seeds 46, 47, 49 on every fold), frozen token-statistic heads on the recipe A fold encoders (spectrum, cov_sqrt, autocorr), '
                    'graph network on flip-TTA / 3-seed-averaged pre-only peaks (4 members). Deploy weights (fit on all five folds): dino 0.125, spectrum 0.25, cov_sqrt 0.375, autocorr 0.25, TTA graph 1.0 (replaces the shipped graph CV). '
                    'File: out/stack5_best_3seeds.npz. The earlier -0.040 (out/stack5_dino_fz_tta.npz) used DINOv3 seed 47 on folds 0 and 4 only: a per-fold member difference, withdrawn.')
s = '\n'.join(lines)
s += '''

## Daily review 2026-11-09 09:00 EST (continue or stop)

| Direction | Decision |
|---|---|
| Graph variants on detector peaks | Stopped 11-08 evening. TTA quartet stays in the lane best. |
| Frozen token-statistic heads | Done, in the lane best. |
| DINOv3 B/16 seeds (label-trained image model, the one lever left per the root-cause read) | Continue: seed 50 training on GPU 2 (folds 0, 1, 4 done); add to the DINOv3 mean when all five folds land. |
| Cross-lane members | ViT-L/16 closed (five folds: alone -0.003, weight 0 in the stack). CPSAM 512 multi-target (HEX lane): read when its folds land. |
| Segmentation features (HEX lane) | Stopped: +0.011 on top of the stack (two runs). |
'''
open(p, 'w').write(s); print('ok')
