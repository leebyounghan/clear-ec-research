p = 'source/cv_stack5_20261108.py'; s = open(p).read()
old = """    v = np.zeros(N_)
    for f in FOLDS:
        g = last5(f'lane_cv_20261108/arms/dinov3_order_f{f}_s46', f)
        if g is None: return None
        v[g[0]] = g[1] - np.log(A[g[0], 1])
    return {f: v for f in FOLDS}"""
assert s.count(old) == 1
new = """    v = np.zeros(N_); seeds = [int(q) for q in os.environ.get('ORDER_SEEDS', '46').split(',')]   # ORDER_SEEDS=46,47: mean of log CV over seeds
    for f in FOLDS:
        got = [q for q in (last5(f'lane_cv_20261108/arms/dinov3_order_f{f}_s{sd}', f) for sd in seeds) if q is not None]
        if len(got) < len(seeds): return None
        r = got[0][0]; assert all(np.array_equal(r, g[0]) for g in got); v[r] = np.mean([g[1] for g in got], 0) - np.log(A[r, 1])
    return {f: v for f in FOLDS}"""
s = s.replace(old, new); open(p, 'w').write(s); print('ok')
