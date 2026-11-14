"""Tier p3 / p3b parity read against p2 (deploy lane, 2026-11-14, D-COORD-P3-PREP). Same frames, settings and switches on both sides (p2_all of the p2
run). Per frame: CD of p2 and the variant and their difference; V (the ViT-L member value) of both; CV / HEX largest |difference| and relative
difference, to be read against the noise measured for p2 (switch configs vs tier p up to CV 5.1e-9 / HEX 1.7e-9; p2 code with switches off vs tier p
CV 2.8e-9 / HEX 1.2e-9); every part that differs at all; the ViT-L stage seconds (vitl_tta_only) and the read bytes per frame; first-frame RSS.
usage: compare_p3_20261114.py VARIANT.json P2_ALL.json [OUT.json]"""
import json, sys
import numpy as np
A, B = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
ra, rb = {r['frame']: r for r in A['rows']}, {r['frame']: r for r in B['rows']}; fr = sorted(set(ra) & set(rb))
REF = B['tier']; o = dict(variant=A['tier'], reference=REF, frames=fr, done=A['done'], env_variant={k: v for k, v in A['env'].items() if v}, env_reference={k: v for k, v in B['env'].items() if v})
o['per_frame'] = {f: dict(CD_p2=rb[f]['CD'], CD_v=ra[f]['CD'], CD_diff=ra[f]['CD'] - rb[f]['CD'], CD_rel=(ra[f]['CD'] - rb[f]['CD']) / rb[f]['CD'],
                          V_p2=rb[f]['parts']['V'][0], V_v=ra[f]['parts']['V'][0], V_rel=(ra[f]['parts']['V'][0] - rb[f]['parts']['V'][0]) / rb[f]['parts']['V'][0],
                          vitl_s_p2=rb[f]['stage_s'].get('vitl_tta_only'), vitl_s_v=ra[f]['stage_s'].get('vitl_tta_only'),
                          read_gb_p2=rb[f]['reads']['read_bytes'] / 1e9, read_gb_v=ra[f]['reads']['read_bytes'] / 1e9, wall_p2=rb[f]['wall'], wall_v=ra[f]['wall']) for f in fr}
for q in ('CD', 'CV', 'HEX'):
    x = np.array([ra[f][q] for f in fr]); y = np.array([rb[f][q] for f in fr])
    o[q] = dict(max_abs=float(np.max(np.abs(x - y))), max_rel=float(np.max(np.abs(x - y) / np.abs(y))), identical=int(np.sum(x == y)))
dp = {}
for f in fr:
    for k in set(ra[f]['parts']) & set(rb[f]['parts']):
        u, v = np.asarray(ra[f]['parts'][k], float), np.asarray(rb[f]['parts'][k], float)
        if u.shape != v.shape: dp[k] = 'shape'; continue
        if not np.array_equal(u, v, equal_nan=True):
            d = np.abs(u - v); m = float(np.max(d[~np.isnan(d)])) if (~np.isnan(d)).any() else float('nan')
            dp[k] = max(dp.get(k, 0.0), m) if not isinstance(dp.get(k), str) else 'shape'
o['parts_differ'] = dict(sorted(dp.items(), key=lambda kv: -kv[1] if isinstance(kv[1], float) else 0))
o['vitl_s_median'] = dict(p2=float(np.median([rb[f]['stage_s'].get('vitl_tta_only', np.nan) for f in fr])), v=float(np.median([ra[f]['stage_s'].get('vitl_tta_only', np.nan) for f in fr])))
o['read_gb_mean'] = dict(p2=float(np.mean([rb[f]['reads']['read_bytes'] for f in fr]) / 1e9), v=float(np.mean([ra[f]['reads']['read_bytes'] for f in fr]) / 1e9))
o['rss_first_gb'] = dict(p2=rb[fr[0]]['ru_maxrss_gb'], v=ra[fr[0]]['ru_maxrss_gb']); o['rss_max_gb'] = dict(p2=max(rb[f]['ru_maxrss_gb'] for f in fr), v=max(ra[f]['ru_maxrss_gb'] for f in fr))
o['wall_median'] = dict(p2=float(np.median([rb[f]['wall'] for f in fr])), v=float(np.median([ra[f]['wall'] for f in fr])))
print(f"{o['variant']} vs {REF} ({sys.argv[2].rsplit('/', 1)[-1]}), {len(fr)} frames, done {o['done']}; keys *_p2 = the reference")
for f, v in o['per_frame'].items():
    print(f"  frame {f}: CD {REF} {v['CD_p2']:.6f} {o['variant']} {v['CD_v']:.6f} diff {v['CD_diff']:+.6f} ({v['CD_rel']*100:+.4f} %)  V {v['V_p2']:.3f} -> {v['V_v']:.3f} "
          f"({v['V_rel']*100:+.3f} %)  vitl s {v['vitl_s_p2']} / {v['vitl_s_v']}  read GB {v['read_gb_p2']:.3f} / {v['read_gb_v']:.3f}")
print('  ' + '  '.join(f"{q} max|d| {o[q]['max_abs']:.3g} rel {o[q]['max_rel']:.3g} same {o[q]['identical']}/{len(fr)}" for q in ('CD', 'CV', 'HEX')))
print('  parts differing:', {k: (f'{v:.3g}' if isinstance(v, float) else v) for k, v in o['parts_differ'].items()})
print(f"  vitl stage median s: {REF} {o['vitl_s_median']['p2']:.2f} {o['variant']} {o['vitl_s_median']['v']:.2f}; read GB/frame {REF} {o['read_gb_mean']['p2']:.3f} {o['variant']} {o['read_gb_mean']['v']:.3f}; "
      f"RSS first {o['rss_first_gb']}; wall median {o['wall_median']}")
if len(sys.argv) > 3: json.dump(o, open(sys.argv[3], 'w'), indent=1)
