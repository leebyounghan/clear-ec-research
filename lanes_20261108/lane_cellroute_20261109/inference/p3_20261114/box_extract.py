# Runs on the box (python3 - V < this): per job of test_<V> and tier p's test_20261113/p, the three outputs, seconds, wall, ram, gpu peak and the
# last stdout line's hooks / decisions / mode / errors; prints one JSON object.
import json, os, sys
V = sys.argv[1]; W = os.path.expanduser('~/workspace/clear-ec')
roots = {V: f'{W}/package_20261114/test_{V}', 'p': f'{W}/package_20261112/test_20261113/p'}
o = {}
for t, R in roots.items():
    for m in ('default', 't4path'):
        if not os.path.isdir(f'{R}/{m}'): continue
        for n in sorted(os.listdir(f'{R}/{m}')):
            d = f'{R}/{m}/{n}'; r = {}
            r['out'] = [json.load(open(f'{d}/output/{k}.json')) if os.path.exists(f'{d}/output/{k}.json') else None for k in ('cell-density', 'coefficient-of-variation', 'hexagonality')]
            try: r['job'] = json.load(open(f'{d}/job.json'))
            except Exception: r['job'] = None
            try: j = json.loads(open(f'{d}/stdout.json').read().strip().splitlines()[-1]); r['last'] = {k: j.get(k) for k in ('seconds', 'setup_s', 'ram_peak_gb', 'gpu', 'gpu_peak_gb', 'hooks', 'decisions', 'mode', 'errors', 'notes')}
            except Exception as e: r['last'] = repr(e)
            o[f'{t}/{m}/{n}'] = r
print(json.dumps(o))
