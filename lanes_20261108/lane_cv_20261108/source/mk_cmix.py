p = 'source/cv_stack5_20261108.py'; s = open(p).read()
if 'def cmix_comp' not in s:
    fn = '''

def cmix_comp(track='raw'):
    """Backbone lane C-Mixup DINOv3-B seed 46 (runs/cmix_f{f}_dinov3b16_s46), one track, log mean of epochs 8-12 (as the backbone lane reads it)."""
    v = np.zeros(N_)
    for f in FOLDS:
        run = X / f'lane_backbone_20261108/runs/cmix_f{f}_dinov3b16_s46/{track}/runs/fold{f}/dinov3_b16'
        if not (run / 'predictions_epoch_12.npz').exists(): return None
        out = []
        for e in range(8, 13):
            with np.load(run / f'predictions_epoch_{e:02d}.npz', allow_pickle=True) as z: out.append((z['indices'].astype(int), z['pred'].astype(np.float64)))
        r = np.array([pos[int(i)] for i in out[0][0]]); v[r] = np.mean([np.log(q[:, 1]) for _, q in out], 0) - np.log(A[r, 1])
    return {f: v for f in FOLDS}


def distdino_comp():'''
    s = s.replace('\n\ndef distdino_comp():', fn, 1)
    s = s.replace("    elif s == 'orderdino': c = orderdino_comp()\n", "    elif s == 'orderdino': c = orderdino_comp()\n    elif s.startswith('cmix'): c = cmix_comp(s.split(':')[1] if ':' in s else 'raw')\n", 1)
    open(p, 'w').write(s)
print('ok', s.count('cmix_comp'))
