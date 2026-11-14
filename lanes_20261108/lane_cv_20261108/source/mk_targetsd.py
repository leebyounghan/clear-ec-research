p = 'source/family_cv_20261108.py'; s = open(p).read()
old = """    with np.load(b.META, allow_pickle=True) as z:
        ids, y = z['ids'].astype(str), z['Y'].astype(np.float32)
    with np.load(SPLITS[fold]) as z:"""
assert s.count(old) == 1
new = """    with np.load(b.META, allow_pickle=True) as z:
        ids, y = z['ids'].astype(str), z['Y'].astype(np.float32)
    if os.environ.get('TARGET_SD'):   # CV lane 11-10: column 1 becomes the cell-area SD (um^2) = CV x 1e6 / CD (CV = SD/mean area, mean area = 1e6/CD); CD and HEX unchanged
        ok = (y[:, 0] > 0) & (y[:, 1] > 0); y = y.copy(); y[ok, 1] = (y[ok, 1].astype(np.float64) * 1e6 / y[ok, 0].astype(np.float64)).astype(np.float32)
    with np.load(SPLITS[fold]) as z:"""
s = s.replace(old, new)
s = s.replace('"""CV lane copy (2026-11-08) of the CD lane runner, adds CD_LOSS=cvw3 (CV column x3).', '"""CV lane copy (2026-11-08) of the CD lane runner, adds CD_LOSS=cvw3 (CV column x3); TARGET_SD=1 (11-10) trains column 1 on the cell-area SD = CV x 1e6 / CD instead of CV.', 1)
open(p, 'w').write(s); print('ok')
