"""Score impact of a precision variant's output deviations (deploy lane, 2026-11-12, D-PKG-004). CPU, seconds.
The test_t4 runs give, per frame, the signed relative change of CD / CV / HEX against the reference run on 20 frames. This read applies such
changes to the tier-t1 strict OOF predictions of all 8,954 rows and reports the change in MAPE (zero labels not scored), so a deviation that
fails the 0.05 % rule can be read in score units. Each row draws a change from the run's 20 observed values (with replacement; 'signed'), or
the same with a random sign ('symmetric', removes the 20-frame mean, which 20 frames estimate poorly). 2,000 draws; mean and 2.5-97.5 %.
Assumption: the change on a frame is independent of that frame's error (not testable on 20 frames). Sanity check first: the OOF arrays must
reproduce the strict t1 values (CD 6.1737, CV 10.1115, HEX 9.71342).
usage: impact_t4.py REF RUN [RUN ...] [--out NAME]"""
import sys, json
import numpy as np
R = '/workspace/clear-ec-research/experiments/'; L = R + 'lane_cellroute_20261109/'; LH = R + 'lane_hex_20261108/'
sys.path[:0] = [LH + 'source', R + 'meta_ceiling_20261103/source', L + 'source', R + 'lane_cv_20261108/source']
import best_pipeline_20261106 as BP
args = sys.argv[1:]; out = None
if '--out' in args: i = args.index('--out'); out = args[i + 1]; del args[i:i + 2]
idx = BP.idx.astype(int); Y = np.asarray(BP.y, float)
zc = np.load(R + 'lane_cd_20261108/reads/cd_tier_t1_20261112.npz'); assert (zc['idx'].astype(int) == idx).all()
zv = np.load(R + 'lane_cv_20261108/deploy_t1/cv_final_oof.npz', allow_pickle=True); assert (zv['idx'].astype(int) == idx).all()
zh = np.load(L + 'out/corr_hex_e2e_strict_a_chain_convention.npz'); assert (zh['idx'].astype(int) == idx).all()
mape = lambda p, y: float(np.mean(np.abs(p[y > 0] / y[y > 0] - 1)) * 100)
cands = dict(CD={'oof_cd': zc['oof_cd']}, CV={'cv': zv['cv'], 'P1': zv['P'][:, 1]}, HEX={'P': zh['P'], 'P_kept': zh['P_kept']})
TARGET = dict(CD=6.1737, CV=10.1115, HEX=9.71342); PRED = {}
for c, k in (('CD', 0), ('CV', 1), ('HEX', 2)):
    got = {n: mape(np.asarray(p, float), Y[:, k]) for n, p in cands[c].items()}
    n = min(got, key=lambda q: abs(got[q] - TARGET[c])); print(c, 'OOF MAPE candidates', got, '-> using', n, flush=True)
    assert abs(got[n] - TARGET[c]) < 6e-4, (c, got); PRED[c] = (np.asarray(cands[c][n], float), Y[:, k])
ref = {r['frame']: r for r in json.load(open(L + f'out/{args[0]}.json'))['rows']}
rng = np.random.default_rng(20261112); report = dict(ref=args[0], base_mape={c: mape(*PRED[c]) for c in PRED}, runs={})
for name in args[1:]:
    rows = [r for r in json.load(open(L + f'out/{name}.json'))['rows'] if r['frame'] in ref]; res = dict(frames=len(rows))
    for mode in ('signed', 'symmetric'):
        dm = {c: [] for c in PRED}
        for b in range(2000):
            for c, (p, y) in PRED.items():
                d = np.array([r[c] / ref[r['frame']][c] - 1 for r in rows]); s = d[rng.integers(0, len(d), len(p))]
                if mode == 'symmetric': s = s * rng.choice([-1.0, 1.0], len(p))
                dm[c].append(mape(p * (1 + s), y) - report['base_mape'][c])
        sc = np.mean([dm[c] for c in PRED], 0)
        res[mode] = {c: dict(mean=float(np.mean(v)), lo=float(np.percentile(v, 2.5)), hi=float(np.percentile(v, 97.5))) for c, v in dm.items()}
        res[mode]['score'] = dict(mean=float(np.mean(sc)), lo=float(np.percentile(sc, 2.5)), hi=float(np.percentile(sc, 97.5)))
    report['runs'][name] = res
    print(name, json.dumps({m: {c: round(v['mean'], 5) for c, v in res[m].items()} for m in ('signed', 'symmetric')}), flush=True)
if out: json.dump(report, open(L + f'out/{out}.json', 'w'), indent=1)
