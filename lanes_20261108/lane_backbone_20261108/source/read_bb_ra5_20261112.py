"""Five-fold member read for the synthetic-pretrained recipe A s46 (synthft_ra_f{0..4}_s46; D-SYN-003 / D-SYN-005), backbone lane 2026-11-12.
Runs read_bb_20261108.main() unchanged, with REF extended to folds 1-3: plain recipe A seeds from oof_recipeA_20261027 (fold 2 seed 46 is
regression_std_s46_v2, the zero-policy rerun; the first s46 attempt failed its CPU gate and has no runs). Seed 48 exists only for folds 0 and 4
(the fold 1-3 s48 runs were cut, seeds_cut_log.json), so 'ref3seeds'/'delta3' on folds 1-3 is the mean of seeds 46 and 47.
Usage: same as read_bb_20261108.py."""
import sys
sys.path.insert(0, '/workspace/clear-ec-research/experiments/lane_backbone_20261108/source')
import read_bb_20261108 as RB  # noqa: E402
O = 'oof_recipeA_20261027'
RB.REF.update({1: {46: f'{O}/fold1/regression_std_s46', 47: f'{O}/fold1/regression_std_s47'},
               2: {46: f'{O}/fold2/regression_std_s46_v2', 47: f'{O}/fold2/regression_std_s47'},
               3: {46: f'{O}/fold3/regression_std_s46', 47: f'{O}/fold3/regression_std_s47'}})
if __name__ == '__main__':
    RB.main()
