"""Real-T4 (Colab) job results against the H200 runs on the same frames (deploy lane, 2026-11-12, D-PKG-006). CPU, seconds.
Reads out/colab_t4c_<mode>.json (downloaded from /content/t4test/out/) and, per mode, compares each job's three outputs with the H200 reference
(t4_ref_20: original fp32 weights, bf16, the lanes' numbers) and with the H200 run of the same precision path where one exists (so the T4's real
kernels are compared both with the validated outputs and with the H200 emulation of the T4 path). Prints a timing / memory table and the
deviation table. usage: colab_compare.py MODE [MODE ...] [--out NAME]"""
import sys, json, os, numpy as np
L = '/workspace/clear-ec-research/experiments/lane_cellroute_20261109/'
args = sys.argv[1:]; out = None
if '--out' in args: i = args.index('--out'); out = args[i + 1]; del args[i:i + 2]
SAME = dict(default='t4_fp16_20', fp16='t4_fp16_20', fp16b1='t4_fp16_b1_20', mix='t4_mix_20', fp32='t4_fp32_20', fp32b1='t4_fp32_b1_20', emu='t4_emu_20', bf16='t4_pkg_20')
load = lambda n: {r['frame']: r for r in json.load(open(L + f'out/{n}.json'))['rows']} if os.path.exists(L + f'out/{n}.json') else None
REF = load('t4_ref_20'); rep = {}
for mode in args:
    J = json.load(open(L + f'out/colab_t4c_{mode}.json'))['jobs']; same = load(SAME.get(mode, '')) if SAME.get(mode) else None
    ok = [j for j in J if j['rc'] == 0 and j['entry_mode'] == 'recipe']
    def dev(refrows):
        if not refrows: return None
        d = np.array([[abs(j['output'][k] / refrows[j['frame']][c] - 1) * 100 for k, c in enumerate(('CD', 'CV', 'HEX'))] for j in ok if j['frame'] in refrows])
        return None if not len(d) else {c: dict(median_pct=float(np.median(d[:, k])), max_pct=float(d[:, k].max())) for k, c in enumerate(('CD', 'CV', 'HEX'))} | {'frames': int(len(d))}
    st = [j['stage_s'] for j in ok if j.get('stage_s')]
    stages = {k: float(np.median([s.get(k, 0) for s in st])) for k in sorted({k for s in st for k in s})} if st else None
    rep[mode] = dict(jobs=len(J), ok=len(ok), rc=[j['rc'] for j in J], wall_s=[j['wall_s'] for j in J], entry_s=[j['entry_seconds'] for j in J],
                     gpu_alloc_peak_gb=max([j['gpu_alloc_peak_gb'] or 0 for j in J]), gpu_device_peak_mib=max([j['gpu_device_peak_mib'] or 0 for j in J]),
                     ram_peak_gb=max([j['ram_peak_gb'] or 0 for j in J]), read_gb=[round((j['hooks'] or {}).get('read_bytes', 0) / 1e9, 1) for j in J],
                     read_s=[round((j['hooks'] or {}).get('read_s', 0), 1) for j in J], autocast=[(j['hooks'] or {}).get('autocast') for j in J][:1],
                     vs_ref=dev(REF), vs_h200_same_path=dev(same), h200_same_path=SAME.get(mode), stage_median_s=stages,
                     errors=[j['errors'] for j in J if j['errors']][:2], stderr=[j['stderr_tail'][-300:] for j in J if j['stderr_tail']][:1])
    r = rep[mode]
    print(f"== {mode}: ok {r['ok']}/{r['jobs']} rc {r['rc']} wall {r['wall_s']} s (read {r['read_s']} s of {r['read_gb']} GB) GPU alloc {r['gpu_alloc_peak_gb']} GB device {r['gpu_device_peak_mib']} MiB RAM {r['ram_peak_gb']} GB autocast {r['autocast']}")
    for k in ('vs_ref', 'vs_h200_same_path'):
        if r[k]: print(f"   {k}: " + '  '.join(f"{c} {r[k][c]['median_pct']:.4f}/{r[k][c]['max_pct']:.3f}%" for c in ('CD', 'CV', 'HEX')) + f"  ({r[k]['frames']} frames)")
    if stages: print('   stages s:', json.dumps({k: round(v, 1) for k, v in stages.items()}))
    if r['errors'] or r['stderr']: print('   errors:', r['errors'], r['stderr'])
if out: json.dump(rep, open(L + f'out/{out}.json', 'w'), indent=1)
