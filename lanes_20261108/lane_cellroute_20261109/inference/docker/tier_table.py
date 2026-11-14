"""Per-tier packaging table (deploy lane, D-PKG-002/003): model archive (files, GB, gzip estimate at the measured ratio 0.93), GPU and RAM peak,
per-job wall (Grand Challenge style jobs where run: full, p; else the in-process steady frame + the measured 5 s start-up), per-job bytes read through
torch.load (all reads incl. repeats) and their page-cache seconds, and cold wall estimates at 150 / 300 / 600 MB/s:
  low  = wall + archive bytes / speed            (every file read from disk once; repeats served from page cache)
  high = wall + per-job bytes read / speed       (every read from disk; page cache does not help)
All timings on an H200 (idle GPU 2); an A10G adds forward time (not measured here). usage: tier_table.py"""
import json, os
L = '/workspace/clear-ec-research/experiments/lane_cellroute_20261109/'; O = L + 'out/'
ld = lambda f: json.load(open(O + f + '.json')) if os.path.exists(O + f + '.json') else None
man = lambda t: json.load(open(L + 'package_20261112/' + ('' if t == 'full' else t + '/') + 'context/MANIFEST.json'))
STRICT = dict(full=dict(CD=6.1743, CV=10.1160, HEX=9.71342), t1=dict(CD=6.1737, CV=10.1115, HEX=9.71342), p=dict(CD=6.1737, CV=10.1115, HEX=9.71943),
              t2=dict(CD=6.1875, CV=10.1115, HEX=9.71943), t3=dict(CD=6.2138, CV=10.1138, HEX=9.73076))
rows = {}
for t in ('full', 'p', 't1', 't2', 't3'):
    m = man(t); inp = ld('packaged_tier_p_20' if t == 'p' else f'packaged_tier_{t}_3')['summary']
    nfr = inp['frames']; rb = inp['hook_stats']['read_bytes'] / nfr / 1e9; rs = inp['hook_stats']['read_s'] / nfr
    gj = ld('gc_jobs_20_unload' if t == 'full' else f'gc_jobs_20_{t}') if t in ('full', 'p') else None
    if gj: wall, src, ram, setup = gj['summary']['wall_median'], '20 GC jobs (median)', gj['summary'].get('ram_peak_gb') or gj['summary'].get('host_maxrss_gb'), gj['summary'].get('setup_s_median') or 5.0
    else: wall, src, ram, setup = inp['wall_median_rest'] + 5.0, 'in-process steady + 5 s start-up', None, 5.0
    if gj and gj['summary'].get('read_gb_median'): rb, rs = gj['summary']['read_gb_median'], gj['summary']['read_s_median']
    cold = {f'{s}MBps': dict(low=round(wall + m['model_gb'] * 1e3 / s), high=round(wall + rb * 1e3 / s)) for s in (150, 300, 600)}
    mean3 = round(sum(STRICT[t].values()) / 3, 4)
    rows[t] = dict(strict=STRICT[t], strict_mean=mean3, model_files=m['model_files'], model_gb=m['model_gb'], model_gz_gb_est=round(m['model_gb'] * 0.93, 1), fp16_files=m['model_fp16_files'],
                   gpu_peak_gb=round(inp['gpu_peak_gb'], 2), ram_peak_gb=ram, wall_s=round(wall, 1), wall_source=src, setup_s=setup, read_gb_per_job=round(rb, 1),
                   read_s_pagecache=round(rs, 1), compute_build_s=round(wall - setup - rs, 1), cold_wall_s=cold, tag=m['tag'])
json.dump(rows, open(L + 'inference/docker/tier_table.json', 'w'), indent=1)
print('| tier | strict CD / CV / HEX (mean) | archive files / GB (gz est) | GPU / RAM peak GB | wall s | read GB/job (s) | compute+build s | cold s @150 / 300 / 600 MB/s |')
print('|---|---|---|---|---|---|---|---|')
for t, r in rows.items():
    s = r['strict']; c = r['cold_wall_s']
    print(f"| {t} | {s['CD']} / {s['CV']} / {s['HEX']} ({r['strict_mean']}) | {r['model_files']} / {r['model_gb']} ({r['model_gz_gb_est']}) | {r['gpu_peak_gb']} / {r['ram_peak_gb'] or '-'} | {r['wall_s']} | {r['read_gb_per_job']} ({r['read_s_pagecache']}) | {r['compute_build_s']} | "
          f"{c['150MBps']['low']}-{c['150MBps']['high']} / {c['300MBps']['low']}-{c['300MBps']['high']} / {c['600MBps']['low']}-{c['600MBps']['high']} |")
