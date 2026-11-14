p = 'source/cv_stack5_20261108.py'; s = open(p).read()
fn = '''

def vith_comp():
    """ViT-H+/16 member as the CD lane reads it (lane_cd_20261108/reads/vith16_s46_member_preds.npz: idx, fold, pred), log CV minus log recipe A TTA."""
    with np.load(X / 'lane_cd_20261108/reads/vith16_s46_member_preds.npz', allow_pickle=True) as z: idx, fz, pr, have = z['idx'].astype(int), z['fold'].astype(int), z['pred'].astype(np.float64), set(int(q) for q in z['folds'])
    if not set(FOLDS) <= have: return None
    v = np.zeros(N_); r = np.array([pos[int(i)] for i in idx]); m = np.isin(fz, list(FOLDS)) & (pr[:, 1] > 0)
    v[r[m]] = np.log(pr[m, 1]) - np.log(A[r[m], 1])
    return {f: v for f in FOLDS}


def win2x_comp():
    """Backbone lane recipe A window member (win2x), raw track, log mean of epochs 8-12 (as the backbone lane reads it)."""
    v = np.zeros(N_)
    for f in FOLDS:
        run = X / f'lane_backbone_20261108/runs/recipeA_win2x_f{f}_s46/raw/runs/fold{f}/cpdino_b8'
        if not (run / 'predictions_epoch_12.npz').exists(): return None
        out = []
        for e in range(8, 13):
            with np.load(run / f'predictions_epoch_{e:02d}.npz', allow_pickle=True) as z: out.append((z['indices'].astype(int), z['pred'].astype(np.float64)))
        r = np.array([pos[int(i)] for i in out[0][0]]); v[r] = np.mean([np.log(q[:, 1]) for _, q in out], 0) - np.log(A[r, 1])
    return {f: v for f in FOLDS}


def tta_files_comp('''
assert s.count('\n\ndef tta_files_comp(') == 1
s = s.replace('\n\ndef tta_files_comp(', fn, 1)
s = s.replace("    elif s == 'mtcps': c = mtcps_comp()\n", "    elif s == 'mtcps': c = mtcps_comp()\n    elif s == 'vith': c = vith_comp()\n    elif s == 'win2x': c = win2x_comp()\n", 1)
s = s.replace('comp_spec: graph:<path-or-lane-run>[+...] | dino | fz:<arm> | g10"""', 'comp_spec: graph:<path-or-lane-run>[+...] | dino | fz:<arm> | g10 | vitl | vith | win2x ..."""', 1)
open(p, 'w').write(s); print('ok', s.count('vith_comp'), s.count('win2x_comp'))
