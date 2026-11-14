p = 'source/cv_stack5_20261108.py'; s = open(p).read()
fn = '''

def sd_comp(cd_source):
    """SD member (TARGET_SD=1, lane_cv arms dinov3_sd_f{f}_s46): CV = SD_pred x CD_pred / 1e6, SD = epochs 16-20 mean log of column 1.
    cd_source 'cdlane': CD lane final OOF CD (lane_cd_20261108/reads/freeze_cd_oof_20261109.npz oof_cd); 'own': the member's own CD head (column 0)."""
    v = np.zeros(N_)
    if cd_source == 'cdlane':
        with np.load(X / 'lane_cd_20261108/reads/freeze_cd_oof_20261109.npz', allow_pickle=True) as z:
            assert np.array_equal(z['idx'].astype(int), BP.idx.astype(int)); logcd = np.log(z['oof_cd'].astype(np.float64))
    for f in FOLDS:
        run = X / f'lane_cv_20261108/arms/dinov3_sd_f{f}_s46' / 'runs' / f'fold{f}' / 'dinov3_b16'
        if not (run / 'predictions_epoch_20.npz').exists(): return None
        out = []
        for e in range(16, 21):
            with np.load(run / f'predictions_epoch_{e:02d}.npz', allow_pickle=True) as z: out.append((z['indices'].astype(int), z['pred'].astype(np.float64)))
        r = np.array([pos[int(i)] for i in out[0][0]]); L = np.mean([np.log(q) for _, q in out], 0)
        lcd = L[:, 0] if cd_source == 'own' else logcd[r]
        v[r] = L[:, 1] + lcd - np.log(1e6) - np.log(A[r, 1])
    return {f: v for f in FOLDS}


def distdino_comp():'''
assert s.count('\\n\\ndef distdino_comp():'.replace('\\\\n', '\\n')) == 1 or s.count('\n\ndef distdino_comp():') == 1
s = s.replace('\n\ndef distdino_comp():', fn, 1)
s = s.replace("    elif s == 'orderdino': c = orderdino_comp()\n", "    elif s == 'orderdino': c = orderdino_comp()\n    elif s in ('sdcd', 'sdown'): c = sd_comp('cdlane' if s == 'sdcd' else 'own')\n", 1)
open(p, 'w').write(s); print('ok', s.count('sd_comp'))
