"""Build-box container tests of one tier against the pod and Colab runs on the same frames (deploy lane, 2026-11-13). Mac, seconds.
Pulls the box jobs (collect_box.py) and, from the pod, the H200 packaged-mode reference of the tier (out/packaged_tier_<tier>_*.json: bf16, fp16
weight copies, the validated outputs) and the real-T4 Colab jobs of the T4 default (fp16 + 1 detector copy). Prints per job: wall, GPU peak,
precision path, outputs, and the relative deviation (%) from the H200 reference, from the Colab T4 job, and (t4path) from the box default.
usage: compare_box.py TIER   -> scratchpad/box/box_test_<tier>_20261113.json"""
import json, subprocess, sys, os
T = sys.argv[1]; S = os.path.dirname(os.path.abspath(__file__)) + '/'; KX = os.path.dirname(S.rstrip('/')) + '/kx'
REF = dict(p='packaged_tier_p_20', t3='packaged_tier_t3_3')[T]; COL = dict(p='colab_t4c6_p_fp16b1', t3='colab_t4c7_t3_fp16b1')[T]
box = json.loads(subprocess.run(['ssh', '-o', 'BatchMode=yes', 'qudgks@210.107.192.239', f'python3 ~/workspace/clear-ec/package_20261112/scripts_20261113/collect_box.py {T}'],
                                capture_output=True, text=True, stdin=subprocess.DEVNULL).stdout)
code = ("import json; O='experiments/lane_cellroute_20261109/out/'; r=json.load(open(O+'%s.json'))['rows']; c=json.load(open(O+'%s.json'))['jobs']; "
        "print(json.dumps(dict(ref={x['frame']: [x['CD'], x['CV'], x['HEX']] for x in r}, colab={x['frame']: x['output'] for x in c if x['rc'] == 0}, "
        "colab_mode={x['frame']: x['entry_mode'] for x in c})))") % (REF, COL)
pod = json.loads(subprocess.run([KX, '60', f'/workspace/venv/bin/python -c "{code}"'], capture_output=True, text=True).stdout.strip().splitlines()[-1])
ref = {int(k): v for k, v in pod['ref'].items()}; col = {int(k): v for k, v in pod['colab'].items() if pod['colab_mode'][k] == 'recipe'}
dev = lambda a, b: [round(abs(x / y - 1) * 100, 4) for x, y in zip(a, b)] if a and b and None not in a else None
dflt = {j['name']: j['output'] for j in box if j['mode'] == 'default'}
rows = []
for j in box:
    parts = j['name'].split('_'); fr = int(parts[1]) if parts[0].isdigit() else None
    r = dict(j, frame=fr, vs_h200_ref=dev(j['output'], ref.get(fr)), vs_colab_t4=dev(j['output'], col.get(fr)),
             vs_box_default=dev(j['output'], dflt.get(j['name'])) if j['mode'] != 'default' else None)
    rows.append(r)
    print(f"{j['mode']:8s} {j['name']:16s} rc {j['rc']} wall {float(j['wall_s'] or 0):6.1f}s entry {j['seconds']}s gpu {j['gpu_device_peak_mib']} MiB "
          f"(alloc {j['gpu_alloc_peak_gb']} GB) ram {j['ram_peak_gb']} GB {j['autocast']}/{j['det_copies']} {j['entry_mode']} out {[round(x, 5) if x else x for x in j['output']]}"
          f" | vsH200 {r['vs_h200_ref']} vsT4 {r['vs_colab_t4']} vsDefault {r['vs_box_default']} {j['errors'] if j['errors'] else ''}")
json.dump(dict(tier=T, h200_reference=REF, colab_t4=COL, jobs=rows), open(S + f'box_test_{T}_20261113.json', 'w'), indent=1)
