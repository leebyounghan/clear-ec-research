p = '/workspace/clear-ec-research/experiments/lane_cv_20261108/source/token_stats_cv_20261108.py'; s = open(p).read()
old = "       3: 'oof_recipeA_20261027/fold3/regression_std_s46', 4: 'phase2_pooling_20261020/arms/std'}"
new = "       3: 'oof_recipeA_20261027/fold3/regression_std_s46', 4: 'phase2_pooling_20261020/arms/std', 99: 'fulldata_recipeA_20261028/regression_std_s46'}  # 99 = full-data recipe A seed 46 (deploy)"
assert old in s; s = s.replace(old, new, 1); open(p, 'w').write(s); print('patched')
