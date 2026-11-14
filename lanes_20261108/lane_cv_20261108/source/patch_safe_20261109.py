p = '/workspace/clear-ec-research/experiments/lane_cv_20261108/source/family_cv_20261108.py'; s = open(p).read()
old = "        ok = y[:, j] > 0; v[k] = float((np.abs(pred[ok, j] - y[ok, j]) / y[ok, j]).mean() * 100)\n    v['mean'] = float(np.mean([v['CD'], v['CV'], v['HEX']])); v['n'] = len(y); return v"
new = "        ok = y[:, j] > 0; v[k] = float((np.abs(pred[ok, j] - y[ok, j]) / y[ok, j]).mean() * 100) if ok.any() else -1.0   # -1 = no scorable rows (full-data split), as in the CD lane runner\n    s = [v[k] for k in ('CD', 'CV', 'HEX') if v[k] >= 0]; v['mean'] = float(np.mean(s)) if s else -1.0; v['n'] = len(y); return v"
assert old in s, 'anchor'; s = s.replace(old, new, 1); open(p, 'w').write(s); print('patched')
