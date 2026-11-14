"""Multi-target CPSAM 512 (HEX lane, 2026-11-08): the family recipe with its own loss (no HEX weighting), plus the fold 1-3
splits that hexcps.py adds. Used for the no-SSL five-fold run after the HEX-weighted loss trailed the backbone on fold 4."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import family_recipe_copy as FR
# folds 1-3 (validation sets checked equal to the chain folds, 2026-11-08)
FR.SPLITS.update({1: 'experiments/ssl_screen_20260911/preflight/fold1.npz', 2: 'experiments/overnight_target_20260912/preflight/fold2.npz',
                  3: 'experiments/overnight_target_20260912/preflight/fold3.npz',
                  99: 'experiments/fulldata_recipeA_20261028/fulldata_regression_split.npz'})   # 11-10: full data (train 8952, the two HEX-zero rows held), as the CD and CV lane runners
if __name__ == '__main__':
    print('mtcps: recipe loss', FR.b.loss_fn.__name__, flush=True)
    FR.main()
