p = 'source/cv_stack5_20261108.py'; s = open(p).read()
fn = '''

def orderdinov2_comp():
    """Order-loss DINOv2 B/14 seed 46 (CD_LOSS=order), lane_cv arms dinov2_order_f{f}_s46, epochs 16-20 mean log CV minus log recipe A TTA."""
    v = np.zeros(N_)
    for f in FOLDS:
        g = last5(f'lane_cv_20261108/arms/dinov2_order_f{f}_s46', f, 'dinov2_b14')
        if g is None: return None
        v[g[0]] = g[1] - np.log(A[g[0], 1])
    return {f: v for f in FOLDS}


def ordervitl_comp():
    """Order-loss ViT-L/16 seed 46 (bb_runner_order_cv_20261110), raw track, log mean of epochs 8-12 (as the ViT-L member is read)."""
    v = np.zeros(N_)
    for f in FOLDS:
        run = X / f'lane_cv_20261108/arms/vitl16_order_f{f}_s46/raw/runs/fold{f}/dinov3_l16t'
        if not (run / 'predictions_epoch_12.npz').exists(): return None
        out = []
        for e in range(8, 13):
            with np.load(run / f'predictions_epoch_{e:02d}.npz', allow_pickle=True) as z: out.append((z['indices'].astype(int), z['pred'].astype(np.float64)))
        r = np.array([pos[int(i)] for i in out[0][0]]); v[r] = np.mean([np.log(q[:, 1]) for _, q in out], 0) - np.log(A[r, 1])
    return {f: v for f in FOLDS}


def distdino_comp():'''
assert s.count('\n\ndef distdino_comp():') == 1
s = s.replace('\n\ndef distdino_comp():', fn, 1)
s = s.replace("    elif s == 'orderdino': c = orderdino_comp()\n", "    elif s == 'orderdino': c = orderdino_comp()\n    elif s == 'orderdinov2': c = orderdinov2_comp()\n    elif s == 'ordervitl': c = ordervitl_comp()\n", 1)
open(p, 'w').write(s); print('stack ok', s.count('orderdinov2_comp'), s.count('ordervitl_comp'))
p = 'source/cv_apply_20261110.py'; s = open(p).read()
old = "'fz:autocorr': ('fz_autocorr', 'fz_std')}"
assert s.count(old) == 1
s = s.replace(old, "'fz:autocorr': ('fz_autocorr', 'fz_std'),\n         'orderdinov2': ('orderdinov2_logcv', 'logA'), 'ordervitl': ('ordervitl_logcv', 'logA')}")
open(p, 'w').write(s); print('apply ok')
