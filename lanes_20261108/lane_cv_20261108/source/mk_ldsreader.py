p = 'source/cv_stack5_20261108.py'; s = open(p).read()
if 'def ldsdino_comp' not in s:
    fn = '''

def ldsdino_comp():
    """LDS DINOv3 B/16 seed 46 (CD_LOSS=lds), lane_cv arms dinov3_lds_f{f}_s46, epochs 16-20 mean log CV minus log recipe A TTA."""
    v = np.zeros(N_)
    for f in FOLDS:
        g = last5(f'lane_cv_20261108/arms/dinov3_lds_f{f}_s46', f)
        if g is None: return None
        v[g[0]] = g[1] - np.log(A[g[0], 1])
    return {f: v for f in FOLDS}


def distdino_comp():'''
    s = s.replace('\n\ndef distdino_comp():', fn, 1)
    s = s.replace("    elif s == 'orderdino': c = orderdino_comp()\n", "    elif s == 'orderdino': c = orderdino_comp()\n    elif s == 'ldsdino': c = ldsdino_comp()\n", 1)
    open(p, 'w').write(s)
print('ok', s.count('ldsdino_comp'))
