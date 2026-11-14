"""Score impact of the T4 default on 200 labelled rows (deploy lane, 2026-11-12, D-GPU-001). CPU, seconds.
Merges the chunk files of run_t4_200_20261112.sh (t4_<name>200_<tier>_c<first>[_partial].json) into out/t4_<name>200_<tier>.json, then reports
  (a) measured: MAPE of the reference run and of the T4 path on the same labelled rows (deploy weights, so both in-sample for the full-data
      models), the change per metric and on the score (mean of the three), paired bootstrap over rows (5,000), 95 % interval;
  (b) the per-row relative deviations (median / max per metric, the 0.05 % rule) and their distribution applied to the strict t1 OOF of all
      8,954 rows as in impact_t4.py (written by calling it with the merged files).
Reading (a): in-sample errors are smaller than OOF errors, so the same perturbation moves in-sample MAPE about as much or more (a random
perturbation costs most where errors are small); (a) is therefore not optimistic. usage: impact200.py TIER [--partial]"""
import sys, os, json, glob, subprocess, numpy as np
R = '/workspace/clear-ec-research/experiments/'; L = R + 'lane_cellroute_20261109/'
sys.path[:0] = [R + 'lane_hex_20261108/source', R + 'meta_ceiling_20261103/source', L + 'source', R + 'lane_cv_20261108/source']
import best_pipeline_20261106 as BP
TIER = sys.argv[1]; PART = '--partial' in sys.argv
idx = BP.idx.astype(int); Y = np.asarray(BP.y, float); row_of = {int(d): k for k, d in enumerate(idx)}


def merged(name):
    rows = {}
    for f in sorted(glob.glob(L + f'out/t4_{name}200_{TIER}_c*.json')):
        if f.endswith('_partial.json') and not PART: continue
        for r in json.load(open(f))['rows']: rows[r['frame']] = r
    if not PART:
        for f in sorted(glob.glob(L + f'out/t4_{name}200_{TIER}_c*_partial.json')):
            for r in json.load(open(f))['rows']: rows.setdefault(r['frame'], r)
    out = L + f'out/t4_{name}200_{TIER}.json'
    json.dump(dict(summary=dict(merged=True, frames=len(rows), gpu_alloc_max_gb=max([r['gpu_alloc_gb'] for r in rows.values()] or [0]),
                                gpu_reserved_max_gb=max([r['gpu_reserved_gb'] for r in rows.values()] or [0]), hook_stats={}), rows=list(rows.values())), open(out, 'w'))
    return rows


ref, t4 = merged('ref'), merged('t4d'); fr = sorted(set(ref) & set(t4)); rep = dict(tier=TIER, frames=len(fr))
k = np.array([row_of[f] for f in fr]); C = ('CD', 'CV', 'HEX')
P_ref = np.array([[ref[f][c] for c in C] for f in fr]); P_t4 = np.array([[t4[f][c] for c in C] for f in fr]); y = Y[k]
ape_ref = np.abs(P_ref / y - 1) * 100; ape_t4 = np.abs(P_t4 / y - 1) * 100; d = ape_t4 - ape_ref
rng = np.random.default_rng(20261112); B = np.array([d[rng.integers(0, len(fr), len(fr))].mean(0) for _ in range(5000)])
rep['measured'] = {c: dict(mape_ref=float(ape_ref[:, j].mean()), mape_t4=float(ape_t4[:, j].mean()), delta=float(d[:, j].mean()),
                          lo=float(np.percentile(B[:, j], 2.5)), hi=float(np.percentile(B[:, j], 97.5))) for j, c in enumerate(C)}
sc = B.mean(1); rep['measured']['score'] = dict(delta=float(d.mean()), lo=float(np.percentile(sc, 2.5)), hi=float(np.percentile(sc, 97.5)))
rel = np.abs(P_t4 / P_ref - 1) * 100
rep['deviation'] = {c: dict(median_pct=float(np.median(rel[:, j])), max_pct=float(rel[:, j].max()), p90_pct=float(np.percentile(rel[:, j], 90))) for j, c in enumerate(C)}
print(json.dumps(rep, indent=1), flush=True)
if len(fr) >= 20:
    o = subprocess.run([sys.executable, L + 'inference/impact_t4.py', f't4_ref200_{TIER}', f't4_t4d200_{TIER}', '--out', f't4_impact200_{TIER}'], capture_output=True, text=True)
    print(o.stdout[-800:], o.stderr[-400:])
    if os.path.exists(L + f'out/t4_impact200_{TIER}.json'): rep['oof_extrapolated'] = json.load(open(L + f'out/t4_impact200_{TIER}.json'))['runs'][f't4_t4d200_{TIER}']
json.dump(rep, open(L + f'out/t4_impact200_{TIER}_report.json', 'w'), indent=1)
