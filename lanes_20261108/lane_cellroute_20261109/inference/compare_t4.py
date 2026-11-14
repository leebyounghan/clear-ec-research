"""Compare test_t4.py runs with a reference run (deploy lane, 2026-11-12, D-PKG-004). CPU only, seconds.
usage: compare_t4.py REF RUN [RUN ...] [--base BASE] [--out NAME]
  per column (CD / CV / HEX): median / max |relative deviation| in percent on the shared frames, frames identical, the 0.05 % median rule;
  per stage: share of detector peak sets identical (md5 of the peaks rounded to 1e-4), mean |peak count change|, segmentation points identical,
  largest relative / absolute part deviations; GPU peak; median seconds per stage and the ratio to BASE (default REF) for each stage."""
import sys, json, numpy as np
L = '/workspace/clear-ec-research/experiments/lane_cellroute_20261109/'
args = sys.argv[1:]; base = out = None
if '--base' in args: i = args.index('--base'); base = args[i + 1]; del args[i:i + 2]
if '--out' in args: i = args.index('--out'); out = args[i + 1]; del args[i:i + 2]
ref_name, runs = args[0], args[1:]; base = base or ref_name
load = lambda n: json.load(open(L + f'out/{n}.json'))
REF = load(ref_name); BASE = load(base); ref = {r['frame']: r for r in REF['rows']}


def peaksets(r):
    s = {}
    if 'peaks' in r:
        s['mean'] = r['peaks']['mean']
        for k, v in enumerate(r['peaks']['tta']): s[f'tta{k}'] = v
        for k, v in enumerate(r['peaks']['single']): s[f'single{k}'] = v
    for d, v in (r.get('other') or {}).items(): s[d] = v
    if 'segpts' in r: s['segpts'] = r['segpts']
    return s


def stage_medians(run):
    keys = sorted({k for r in run['rows'] for k in r['stage_s']})
    return {k: float(np.median([r['stage_s'].get(k, 0.0) for r in run['rows']])) for k in keys}


bmed = stage_medians(BASE); report = dict(ref=ref_name, base=base, runs={})
for name in runs:
    run = load(name); rr = [r for r in run['rows'] if r['frame'] in ref]; res = dict(frames=len(rr), summary_env=run['summary'].get('env'))
    for c in ('CD', 'CV', 'HEX'):
        a = np.array([abs(r[c] / ref[r['frame']][c] - 1) * 100 for r in rr])
        res[c] = dict(median_pct=float(np.median(a)), max_pct=float(a.max()), identical=int((a == 0).sum()))
    res['rule_median_lt_0.05pct'] = all(res[c]['median_pct'] < 0.05 for c in ('CD', 'CV', 'HEX'))
    same = tot = 0; dn = []; per = {}
    for r in rr:
        a, b = peaksets(r), peaksets(ref[r['frame']])
        for k in b:
            if k not in a: continue
            eq = a[k]['md5'] == b[k]['md5']; same += eq; tot += 1; dn.append(abs(a[k]['n'] - b[k]['n']))
            g = 'segpts' if k == 'segpts' else ('pre-only' if k in ('mean',) or k[:3] in ('tta', 'sin') else 'other13')
            per.setdefault(g, [0, 0, []]); per[g][0] += eq; per[g][1] += 1; per[g][2].append(abs(a[k]['n'] - b[k]['n']) / max(b[k]['n'], 1) * 100)
    res['peak_sets_identical'] = f'{same}/{tot}'
    res['peaks_by_group'] = {g: dict(identical=f'{v[0]}/{v[1]}', count_change_pct_median=float(np.median(v[2])), count_change_pct_max=float(np.max(v[2]))) for g, v in per.items()}
    pd = {}
    for r in rr:
        p, q = r['parts'], ref[r['frame']]['parts']
        for k in q:
            if k in p and len(p[k]) == len(q[k]):
                x, y = np.asarray(p[k]), np.asarray(q[k]); ad = np.abs(x - y)
                rd = np.where(np.abs(y) > 1e-6, ad / np.maximum(np.abs(y), 1e-12), 0.0)
                e = pd.setdefault(k, [[], []]); e[0].append(float(ad.max())); e[1].append(float(rd.max()) * 100)
    res['parts'] = {k: dict(abs_median=float(np.median(v[0])), abs_max=float(np.max(v[0])), rel_pct_median=float(np.median(v[1])), rel_pct_max=float(np.max(v[1])))
                    for k, v in sorted(pd.items(), key=lambda kv: -np.median(kv[1][1]))}
    res['gpu_alloc_max_gb'] = run['summary']['gpu_alloc_max_gb']; res['gpu_reserved_max_gb'] = run['summary']['gpu_reserved_max_gb']
    med = stage_medians(run); res['wall_median'] = float(np.median([r['wall'] for r in run['rows']])); res['wall_median_base'] = float(np.median([r['wall'] for r in BASE['rows']]))
    res['stage_median_s'] = {k: [round(v, 2), round(bmed.get(k, float('nan')), 2), round(v / bmed[k], 2) if bmed.get(k) else None] for k, v in med.items()}
    hs = run['summary']['hook_stats']; res['hooks'] = {k: hs.get(k) for k in ('prec', 'autocast', 'no_tf32', 'tf32_refused', 'fp32_modules', 'emu_ops', 'bf16_native_ops', 'autocast_calls')}
    res['tf32_end'] = rr[-1]['tf32'] if rr else None
    report['runs'][name] = res
    print(f"== {name} vs {ref_name} ({len(rr)} frames): " + '  '.join(f"{c} {res[c]['median_pct']:.4f}/{res[c]['max_pct']:.3f}% id {res[c]['identical']}" for c in ('CD', 'CV', 'HEX'))
          + f"  rule {'PASS' if res['rule_median_lt_0.05pct'] else 'FAIL'}  peak sets identical {res['peak_sets_identical']}  GPU {res['gpu_alloc_max_gb']:.2f}/{res['gpu_reserved_max_gb']:.2f} GB"
          + f"  wall {res['wall_median']:.1f} s vs {base} {res['wall_median_base']:.1f} s")
    for g, v in res['peaks_by_group'].items(): print(f'   peaks {g}: {v}')
    for k, v in list(res['parts'].items())[:12]: print(f'   part {k}: rel {v["rel_pct_median"]:.4f}/{v["rel_pct_max"]:.3f}%  abs {v["abs_median"]:.3g}/{v["abs_max"]:.3g}')
    print('   stages (run s, base s, ratio):', json.dumps(res['stage_median_s']))
    print('   hooks:', json.dumps(res['hooks'])[:1500], 'tf32_end', res['tf32_end'])
if out: json.dump(report, open(L + f'out/{out}.json', 'w'), indent=1)
