# bb_runner fixed for fold 2 (backbone lane, 2026-11-10 02:20 EST)

Fold 2's split has two validation rows with HEX = 0. bb_runner_20261108.py asserted positive labels on all rows and stopped at startup; nothing was written. Fixed: training labels must still be positive, the two rows stay in validation (they are predicted, so the OOF covers them), and each per-epoch score leaves a zero label out of that metric, as the official metric does. Folds 0, 1, 3, 4 behave exactly as before (CPU-checked: same scores on fold 0).

Restart recipe A win2x fold 2 with the same command as before (F = 2), once. Root: experiments/lane_backbone_20261108/runs/recipeA_win2x_f2_s46. Check that the root does not exist before starting.

If no lane has started fold 2 by 07:30 EST 11-10 (that root does not exist), the backbone lane starts it on GPU 3. Whoever starts it creates the root at launch, so check the root first.

**Update 06:02 EST 11-10: the backbone lane started fold 2 on GPU 3 (slot B). Do not start it anywhere else.**
