"""Fold-0 seed-mean member read for synthetic-pretrained recipe A (D-SYN-007), backbone lane 2026-11-13.
Synthetic s46 (runs/synthft_ra_f0_s46) and s47 (runs/synthft_ra_f0_s47; same synthetic pretraining, only the fine-tune seed differs)
against plain recipe A fold 0 s46 / s47 / s48 (read_bb_20261108.REF[0]). Member read as read_bb_20261108: per-row APE averaged over the
epoch window, raw track, official score; seed mean = row APE averaged over seeds; paired donor bootstrap (eval_oof_20261031).
Writes reads/synthft_ra_f0_seedmean_w{lo},{hi}.json. Usage: read_bb_seedmean_f0_20261113.py [--window 8,12]"""
import json, sys
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/lane_backbone_20261108/source')
import read_bb_20261108 as RB  # noqa: E402
E, K = RB.E, RB.K
arg = lambda k, d=None: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
lo, hi = [int(v) for v in arg('--window', '8,12').split(',')]
b = E.load_base(); gp = E.donors(b['ids']); pos = {int(i): n for n, i in enumerate(b['idx'])}
runs = {'syn46': 'lane_backbone_20261108/runs/synthft_ra_f0_s46/raw', 'syn47': 'lane_backbone_20261108/runs/synthft_ra_f0_s47/raw',
        **{f'plain{s}': r for s, r in RB.REF[0].items()}}
P = {k: RB.preds(r, 0, lo, hi)[0] for k, r in runs.items()}
ix, y = P['syn46'][0][0], P['syn46'][0][2]
for k, v in P.items():
    assert v is not None and np.array_equal(v[0][0], ix) and np.array_equal(v[0][2], y), k
keep = np.array([int(i) in pos for i in ix]); y = y[keep]; g = gp[np.array([pos[int(i)] for i in ix[keep]])]
mem = {k: np.mean([E.ape(p[keep], y) for _, p, _ in v], 0) for k, v in P.items()}
sc = lambda e: {k: round(v, 4) for k, v in E.official(e).items() if k in K + ['mean']}
def cmp(a, b_):
    pb = E.paired_bootstrap(a, b_, g); return dict(delta=round(pb['delta'], 4), ci95=[round(v, 4) for v in pb['ci95']])
syn2 = (mem['syn46'] + mem['syn47']) / 2; pl2 = (mem['plain46'] + mem['plain47']) / 2; pl3 = (mem['plain46'] + mem['plain47'] + mem['plain48']) / 3
res = dict(window=[lo, hi], n=int(keep.sum()), scores={k: sc(v) for k, v in mem.items()} | {'syn_mean46_47': sc(syn2), 'plain_mean46_47': sc(pl2), 'plain_mean46_48': sc(pl3)},
           syn47_vs_plain47=cmp(mem['syn47'], mem['plain47']), syn47_vs_plain46=cmp(mem['syn47'], mem['plain46']),
           seedmean_syn46_47_vs_plain46_47=cmp(syn2, pl2), seedmean_syn46_47_vs_plain46_48=cmp(syn2, pl3),
           seed_noise_plain47_vs_plain46=cmp(mem['plain47'], mem['plain46']), seed_noise_syn47_vs_syn46=cmp(mem['syn47'], mem['syn46']))
print(json.dumps(res), flush=True)
(RB.LANE / 'reads' / f'synthft_ra_f0_seedmean_w{lo},{hi}.json').write_text(json.dumps(res, indent=1))
