"""One evaluation harness for every OOF candidate (2026-10-31).

Usage:
  python -I eval_oof_20261031.py --pred <file.npz>[:key] [--ref B] [--folds 0,1,2,3] [--out report.json]
  --pred: npz with 'indices' (rows of the 8,954-image cache) and an (N,3) prediction array under `key`
          (default 'pred'); or one of the built-ins A / P1 / B from the hybrid OOF.
Reports: official score (equal-weight mean of per-metric MAPE, zero GT unscored per metric), per metric, per fold,
paired delta vs the reference with a donor-cluster bootstrap (slide_id = donor, folds are donor-grouped),
and error by slice (year, GT tertiles, image statistics, A-vs-P1 disagreement).
Protocol: develop on folds 0-3, confirm once on fold 4 with a threshold written before looking.
"""
import argparse, csv, json, re
from pathlib import Path
import numpy as np

ROOT = Path('/workspace/clear-ec-research')
HYB = ROOT / 'experiments/recipeA_analysis_20261028/hybrid_oof.npz'
STATS = ROOT / 'experiments/eval_harness_20261031/image_stats.npz'
K = ['CD', 'CV', 'HEX']


def load_base():
    z = np.load(HYB, allow_pickle=True)
    return dict(idx=z['indices'].astype(int), ids=np.array([str(i).strip() for i in z['ids']]), y=z['target'].astype(float),
                fold=z['fold'].astype(int), A=z['A_recipeA_tta_scalar'].astype(float), P1=z['historical_five_calibrated'].astype(float),
                B=z['B_hybrid_w0p5'].astype(float))


def load_pred(spec, base):
    if spec in ('A', 'P1', 'B'):
        return base[spec]
    path, _, key = spec.partition(':')
    z = np.load(path, allow_pickle=True)
    pos = {int(i): n for n, i in enumerate(base['idx'])}
    p = np.full((len(base['idx']), 3), np.nan)
    for i, v in zip(z['indices'].astype(int), z[key or 'pred']):
        if int(i) in pos:
            p[pos[int(i)]] = v
    return p


def ape(p, y):
    s = y > 0
    return np.where(s, np.abs(p - y) / np.where(s, y, 1.0) * 100, np.nan)


def official(e, rows=None):
    e = e if rows is None else e[rows]
    m = np.nanmean(e, 0)
    return dict(zip(K, map(float, m)), mean=float(m.mean()), n=int(len(e)))


def donors(ids):
    gt = {re.sub(r'\s+', ' ', r['ID'].strip()): r['slide_id'].strip()
          for r in csv.DictReader(open(ROOT / 'datasets/clear_ec_official/final_train_ids.csv'))}
    d = np.array([gt.get(re.sub(r'\s+', ' ', i), i) for i in ids])
    _, inv = np.unique(d, return_inverse=True)
    return inv


def paired_bootstrap(e1, e0, groups, reps=2000, seed=0):
    rng = np.random.default_rng(seed)
    G = groups.max() + 1
    def sums(e):
        ok = ~np.isnan(e)
        S = np.zeros((G, 3)); C = np.zeros((G, 3))
        for j in range(3):
            np.add.at(S[:, j], groups[ok[:, j]], e[ok[:, j], j]); np.add.at(C[:, j], groups[ok[:, j]], 1)
        return S, C
    S1, C1 = sums(e1); S0, C0 = sums(e0)
    w = rng.multinomial(G, np.full(G, 1 / G), size=reps).astype(float)       # reps x G
    m1 = (w @ S1) / (w @ C1); m0 = (w @ S0) / (w @ C0)
    d = m1.mean(1) - m0.mean(1)
    point = float(np.mean(S1.sum(0) / C1.sum(0)) - np.mean(S0.sum(0) / C0.sum(0)))
    return dict(delta=point, ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], p_better=float(np.mean(d < 0)))


def year_of(i):
    m = re.match(r'^(\d{4})-\d{4}', i)
    if m:
        return int(m.group(1))
    m = re.match(r'^\d{4}-(\d{2})$', i)
    return 2000 + int(m.group(1)) if m else -1


def image_stats(base):
    if STATS.exists():
        z = np.load(STATS)
        if (z['idx'] == base['idx']).all():
            return {k: z[k] for k in z.files if k != 'idx'}
    mm = np.memmap(ROOT / 'datasets/clear_ec_fullframe_20260904/img_full_raw.dat', dtype=np.uint8, mode='r', shape=(8954, 972, 1296))
    out = {k: np.zeros(len(base['idx'])) for k in ('mean', 'contrast', 'black_frac', 'focus')}
    for n, i in enumerate(base['idx']):
        a = np.asarray(mm[i, ::2, ::2], dtype=np.float32)
        out['mean'][n] = a.mean(); out['contrast'][n] = a.std(); out['black_frac'][n] = np.mean(a < 5)
        lap = a[1:-1, 1:-1] * 4 - a[:-2, 1:-1] - a[2:, 1:-1] - a[1:-1, :-2] - a[1:-1, 2:]
        out['focus'][n] = lap.var()
    STATS.parent.mkdir(parents=True, exist_ok=True)
    np.savez(STATS, idx=base['idx'], **out)
    return out


def slices(e_pred, e_ref, base, rows):
    feats = dict(year=np.array([year_of(i) for i in base['ids']]))
    for j, k in enumerate(K):
        feats[f'gt_{k}'] = base['y'][:, j]
    st = image_stats(base)
    feats.update({f'img_{k}': v for k, v in st.items()})
    feats['disagree_A_P1'] = np.abs(np.log(base['A'] / base['P1'])).mean(1)
    rep = {}
    for name, v in feats.items():
        vr = v[rows]
        if name == 'year':
            bins = [(int(y), vr == y) for y in np.unique(vr)]
        else:
            q = np.quantile(vr, [0, 0.25, 0.5, 0.75, 1])
            bins = [(f'q{b + 1} [{q[b]:.3g},{q[b + 1]:.3g}]', (vr >= q[b]) & ((vr < q[b + 1]) if b < 3 else (vr <= q[b + 1]))) for b in range(4)]
        rep[name] = [dict(bin=str(b), n=int(m.sum()), pred=round(official(e_pred[rows][m])['mean'], 3),
                          ref=round(official(e_ref[rows][m])['mean'], 3)) for b, m in bins if m.sum() >= 30]
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', required=True); ap.add_argument('--ref', default='B')
    ap.add_argument('--folds', default='0,1,2,3,4'); ap.add_argument('--out'); ap.add_argument('--no-slices', action='store_true')
    a = ap.parse_args()
    base = load_base()
    folds = [int(f) for f in a.folds.split(',')]
    p, r = load_pred(a.pred, base), load_pred(a.ref, base)
    rows = np.isin(base['fold'], folds) & ~np.isnan(p).any(1)
    e_p, e_r = ape(p, base['y']), ape(r, base['y'])
    g = donors(base['ids'])
    res = dict(pred=a.pred, ref=a.ref, folds=folds, rows=int(rows.sum()),
               score=official(e_p, rows), ref_score=official(e_r, rows),
               per_fold={int(f): round(official(e_p, rows & (base['fold'] == f))['mean'], 4) for f in folds},
               paired_vs_ref=paired_bootstrap(e_p[rows], e_r[rows], np.unique(g[rows], return_inverse=True)[1]))
    if not a.no_slices:
        res['slices'] = slices(e_p, e_r, base, rows)
    s = json.dumps(res, indent=1)
    if a.out:
        Path(a.out).write_text(s)
    print(json.dumps({k: res[k] for k in ('pred', 'ref', 'rows', 'score', 'ref_score', 'per_fold', 'paired_vs_ref')}, indent=1))


if __name__ == '__main__':
    main()
