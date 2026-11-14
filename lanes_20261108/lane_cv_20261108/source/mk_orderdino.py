p = 'source/cv_stack5_20261108.py'; s = open(p).read()
fn = '''

def orderdino_comp():
    """Order-loss DINOv3 B/16 (CD_LOSS=order, seed 46), lane_cv arms dinov3_order_f{f}_s46, epochs 16-20 mean log CV minus log recipe A TTA."""
    v = np.zeros(N_)
    for f in FOLDS:
        g = last5(f'lane_cv_20261108/arms/dinov3_order_f{f}_s46', f)
        if g is None: return None
        v[g[0]] = g[1] - np.log(A[g[0], 1])
    return {f: v for f in FOLDS}


def distdino_comp():'''
assert s.count('\n\ndef distdino_comp():') == 1
s = s.replace('\n\ndef distdino_comp():', fn, 1)
old = "        if os.environ.get('DINO_PLUS_AUG'): roots = {f: roots[f] + [f'lane_cv_20261108/arms/dinov3_aug_f{f}_s{sd}' for sd in (46, 47)] for f in range(5)}\n"
assert s.count(old) == 1
s = s.replace(old, old + "        if os.environ.get('DINO_PLUS_ORDER'): roots = {f: roots[f] + [f'lane_cv_20261108/arms/dinov3_order_f{f}_s46'] for f in range(5)}\n")
s = s.replace("    elif s == 'distdino': c = distdino_comp()\n", "    elif s == 'distdino': c = distdino_comp()\n    elif s == 'orderdino': c = orderdino_comp()\n", 1)
open(p, 'w').write(s); print('ok', s.count('orderdino_comp'), s.count('DINO_PLUS_ORDER'))
