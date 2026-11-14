"""DINOv3 ViT-B/16 stride 16 by epoch window, seeds 46 and 47, folds 0 and 4 (lane GPU 2, 2026-11-07).
EXPLORATORY: the windows were looked at after the validation curves were seen to bottom out at epochs 9-13, so nothing here can pass a gate.
Same added formula as read_two_seeds: H x exp(w (log new - log recipe A TTA)), stretch from best_pipeline. Writes reads/dinov3_b16_epoch_windows.json."""
import json, multiprocessing as mp, sys
from pathlib import Path
import numpy as np
X = Path('/workspace/clear-ec-research/experiments')
sys.path.insert(0, str(X / 'eval_harness_20261031/source')); sys.path.insert(0, str(X / 'meta_ceiling_20261103/source'))
import eval_oof_20261031 as E
import best_pipeline_20261106 as BP
K = ['CD', 'CV', 'HEX']; FOLDS = (0, 4); FAM = 'dinov3_b16'
ROOTS = {'s46': 'lane_gpu3_20261106/runs/dinov3_b16_f{f}_s46', 's47': 'lane_gpu2_20261106/runs/dinov3_b16_f{f}_s47'}
WINDOWS = [(5, 9), (7, 11), (9, 13), (11, 15), (13, 17), (16, 20)]
b = E.load_base(); gp = E.donors(b['ids']); pos = {int(i): n for n, i in enumerate(b['idx'])}; assert np.array_equal(BP.idx, b['idx'])
REFROWS = None


def window(root, f, lo, hi):
    run = X / root.format(f=f) / 'runs' / f'fold{f}' / FAM; out = []
    for e in range(lo, hi + 1):
        with np.load(run / f'predictions_epoch_{e:02d}.npz', allow_pickle=True) as z: out.append((z['indices'].astype(int), z['pred'].astype(np.float64)))
    ix = out[0][0]; assert all(np.array_equal(ix, o[0]) for o in out)
    return np.array([pos[int(i)] for i in ix]), np.mean([np.log(p) for _, p in out], 0)


def seed_window(root, lo, hi):
    R, L = [], []
    for f in FOLDS:
        r, l = window(root, f, lo, hi); R.append(r); L.append(l)
    return np.concatenate(R), np.concatenate(L)


def added(rows, lg):
    A = BP._b['A']; o = {}
    for w in (0.125, 0.25):
        c = np.zeros_like(BP.H); c[rows] = w * (lg - np.log(A[rows])); P = BP.pipe(BP.add(BP.BEST, {ff: c for ff in range(5)})); eb, er = E.ape(P[rows], BP.y[rows]), E.ape(BP.REF_BEST[rows], BP.y[rows]); pb = E.paired_bootstrap(eb, er, gp[rows]); s1, s0 = E.official(eb), E.official(er)
        o[str(w)] = dict(delta=round(pb['delta'], 4), ci95=[round(v, 4) for v in pb['ci95']], per={k: round(s1[k] - s0[k], 4) for k in K})
    return o


def one(win):
    lo, hi = win; out = dict(window=f'{lo}-{hi}'); ls = {}
    for s, root in ROOTS.items():
        rows, lg = seed_window(root, lo, hi); ls[s] = lg
        member = E.official(E.ape(np.exp(lg), BP.y[rows]))
        out[s] = dict(member=round(member['mean'], 4), member_per={k: round(member[k], 4) for k in K}, added=added(rows, lg))
    both = (ls['s46'] + ls['s47']) / 2; member = E.official(E.ape(np.exp(both), BP.y[rows]))
    out['two_seeds'] = dict(member=round(member['mean'], 4), added=added(rows, both))
    return out


if __name__ == '__main__':
    rows, _ = seed_window(ROOTS['s46'], 16, 20)
    A = E.official(E.ape(BP._b['A'][rows], BP.y[rows]))
    with mp.get_context('fork').Pool(4) as p: res = p.map(one, WINDOWS)
    d = dict(name='dinov3_b16_epoch_windows', exploratory=True, why='windows chosen after the curves were seen; cannot pass a gate', folds=list(FOLDS), recipe_a_tta_on_rows=round(A['mean'], 4), best_on_rows=round(E.official(E.ape(BP.REF_BEST[rows], BP.y[rows]))['mean'], 4), windows=res)
    (X / 'lane_gpu2_20261106/reads/dinov3_b16_epoch_windows.json').write_text(json.dumps(d, indent=1))
    print('recipe A TTA on rows', d['recipe_a_tta_on_rows'], 'best on rows', d['best_on_rows'])
    for r in res:
        print(r['window'], ' '.join(f"{s}: member {r[s]['member']:.3f} w.125 {r[s]['added']['0.125']['delta']:+.4f} {r[s]['added']['0.125']['ci95']} w.25 {r[s]['added']['0.25']['delta']:+.4f} {r[s]['added']['0.25']['ci95']} |" for s in ('s46', 's47', 'two_seeds')))
