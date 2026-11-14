p = '/workspace/clear-ec-research/experiments/lane_cv_20261108/source/family_cv_20261108.py'; s = open(p).read()
old = "          4: 'experiments/target_fold4_20260912/preflight/fold4.npz'}"
new = "          4: 'experiments/target_fold4_20260912/preflight/fold4.npz',\n          99: 'experiments/fulldata_recipeA_20261028/fulldata_regression_split.npz'}  # full data as in the CD lane runner (train 8952, the two HEX-zero rows held)"
assert old in s and "99: 'experiments/fulldata" not in s; s = s.replace(old, new, 1); open(p, 'w').write(s); print('patched')
