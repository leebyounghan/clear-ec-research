# CD lane notes (GPU 0), 2026-11-08

Pod folder `experiments/lane_cd_20261108/` (RESULTS.tsv, REPORT.md, reads/, logs/). Times Eastern.

## Finding so far (CPU, stored predictions)

- Other encoder families are worth more on CD than the three-metric reads suggested. On folds 0 and 4, cross-fit weights and stretch:
  - A/P1 geometric reweight for CD alone (w_A ~0.65): -0.035 vs B (5 folds nested: -0.017, CI -0.033, -0.001).
  - + DINOv3 B/16 (s46, s47) + ConvNeXt B + DINOv2 B, epochs 16-20 log mean, about 25 % share: CD 6.185 vs 6.263, -0.079 (CI -0.125, -0.035); -0.044 vs the reweight alone (CI -0.082, -0.007). ConvNeXt carries most; DINOv2 adds nothing.
- Closed as tested: graph/sim CD column (0 to +0.010), per-source blend weights (worse than global), detector density s48 top-3 as a 4th blend component (weight 0), per-source stretch (-0.009).
- ConvNeXt's CD value is flat from epoch 8 to 20 although its own CD is best near epoch 9; 20-epoch schedule kept.

## GPU 0 queue

`experiments/lane_cd_20261108/queue.txt`, runner `source/queue_cd_20261108.sh` (pid 137568), runner copy `source/family_cd_20261108.py` (lane GPU 3's family runner with all five folds wired and fold 2's HEX-zero rows handled). ConvNeXt f1-3, ConvNeXt with CD x3 loss weight on f0/f4, DINOv3 s46 f1-3, DINOv3 s47 f1-3. Started after the old lf25 fold-4 job (left to finish, ~03:50 UTC).

These members also carry CV and HEX predictions; other lanes may read them.
