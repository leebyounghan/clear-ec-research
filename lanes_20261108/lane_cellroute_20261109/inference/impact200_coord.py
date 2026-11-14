"""coordinator 2026-11-13 (D-PKG-007): impact200.py generalised to another precision path NAME on the same 200 rows, tier p.
(a) deviations of NAME from the reference; (b) OOF-applied impact through impact_t4.py; (c) paired on the same rows: NAME vs the fp16 T4
default (t4d): measured MAPE change, and the difference in per-row absolute deviation from the reference. usage: impact200_coord.py NAME"""
import sys, os, json, glob, subprocess, numpy as np
R = '/workspace/clear-ec-research/experiments/'; L = R + 'lane_cellroute_20261109/'
sys.path[:0] = [R + 'lane_hex_20261108/source', R + 'meta_ceiling_20261103/source', L + 'source', R + 'lane_cv_20261108/source']
import best_pipeline_20261106 as BP
NAME = sys.argv[1]; TIER = 'p'
idx = BP.idx.astype(int); Y = np.asarray(BP.y, float); row_of = {int(d): k for k, d in enumerate(idx)}
rows = {}
for f in sorted(glob.glob(L + f'out/t4_{NAME}200_{TIER}_c*.json')):
    if f.endswith('_partial.json'): continue
    for r in json.load(open(f))['rows']: rows[r['frame']] = r
json.dump(dict(summary=dict(merged=True, frames=len(rows), hook_stats={}), rows=list(rows.values())), open(L + f'out/t4_{NAME}200_{TIER}.json', 'w'))
ld = lambda n: {r['frame']: r for r in json.load(open(L + f'out/t4_{n}200_{TIER}.json'))['rows']}
ref, t4d, new = ld('ref'), ld('t4d'), rows; fr = sorted(set(ref) & set(t4d) & set(new)); C = ('CD', 'CV', 'HEX')
k = np.array([row_of[f] for f in fr]); y = Y[k]
P = {n: np.array([[d[f][c] for c in C] for f in fr]) for n, d in (('ref', ref), ('t4d', t4d), ('new', new))}
ape = {n: np.abs(P[n] / y - 1) * 100 for n in P}; rel = {n: np.abs(P[n] / P['ref'] - 1) * 100 for n in ('t4d', 'new')}
rng = np.random.default_rng(20261113); bi = [rng.integers(0, len(fr), len(fr)) for _ in range(5000)]
def ci(d):
    B = np.array([d[i].mean(0) for i in bi]); out = {c: [float(d[:, j].mean()), float(np.percentile(B[:, j], 2.5)), float(np.percentile(B[:, j], 97.5))] for j, c in enumerate(C)}
    s = B.mean(1); out['score'] = [float(d.mean()), float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]; return out
rep = dict(name=NAME, frames=len(fr))
rep['deviation_new'] = {c: dict(median=float(np.median(rel['new'][:, j])), p90=float(np.percentile(rel['new'][:, j], 90)), max=float(rel['new'][:, j].max())) for j, c in enumerate(C)}
rep['deviation_t4d'] = {c: dict(median=float(np.median(rel['t4d'][:, j])), p90=float(np.percentile(rel['t4d'][:, j], 90)), max=float(rel['t4d'][:, j].max())) for j, c in enumerate(C)}
rep['measured_new_minus_ref'] = ci(ape['new'] - ape['ref']); rep['measured_new_minus_t4d'] = ci(ape['new'] - ape['t4d'])
rep['absdev_new_minus_t4d_pct'] = ci(rel['new'] - rel['t4d'])
o = subprocess.run([sys.executable, L + 'inference/impact_t4.py', f't4_ref200_{TIER}', f't4_{NAME}200_{TIER}', f't4_t4d200_{TIER}', '--out', f't4_impact200_{TIER}_{NAME}'], capture_output=True, text=True)
print(o.stdout[-600:], o.stderr[-600:])
p = L + f'out/t4_impact200_{TIER}_{NAME}.json'
if os.path.exists(p):
    runs = json.load(open(p))['runs']; rep['oof_extrapolated'] = {n: {m: [round(v['mean'], 4), round(v['lo'], 4), round(v['hi'], 4)] for m, v in r['symmetric'].items()} for n, r in runs.items()}
json.dump(rep, open(L + f'out/t4_impact200_{TIER}_{NAME}_report.json', 'w'), indent=1)
print(json.dumps(rep, indent=1))
