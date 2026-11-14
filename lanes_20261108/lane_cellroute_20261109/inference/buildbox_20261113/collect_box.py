"""Run on the build box: dump every container-test job of one tier (test_20261113/<tier>/<mode>/<name>) as one JSON line.
usage: python3 collect_box.py TIER"""
import json, os, sys, glob
T = sys.argv[1]; R = os.path.expanduser(f'~/workspace/clear-ec/package_20261112/test_20261113/{T}')
out = []
for d in sorted(glob.glob(R + '/*/*')):
    mode, name = d.split('/')[-2:]
    j = json.load(open(d + '/job.json')) if os.path.exists(d + '/job.json') else {}
    o = []
    for f in ('cell-density', 'coefficient-of-variation', 'hexagonality'):
        try: o.append(json.load(open(f'{d}/output/{f}.json')))
        except Exception: o.append(None)
    try: s = json.loads(open(d + '/stdout.json').read().strip().splitlines()[-1])
    except Exception: s = {}
    h = s.get('hooks') or {}
    out.append(dict(mode=mode, name=name, rc=j.get('rc'), wall_s=j.get('wall_s'), gpu_device_peak_mib=j.get('gpu_device_peak_mib'), output=o,
                    entry_mode=s.get('mode'), seconds=s.get('seconds'), setup_s=s.get('setup_s'), ram_peak_gb=s.get('ram_peak_gb'),
                    gpu_alloc_peak_gb=s.get('gpu_peak_gb'), autocast=h.get('autocast'), det_copies=h.get('det_copies'), read_s=h.get('read_s'),
                    read_gb=(h.get('read_bytes') or 0) / 1e9, errors=s.get('errors'), gpu=s.get('gpu'), decisions=s.get('decisions')))
print(json.dumps(out))
