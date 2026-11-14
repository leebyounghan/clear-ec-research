"""Dry check of test_budget.py's file selection for the -nodet / -cdonly variants on the fp32 run's file list (no GPU)."""
import json, os, re, collections
E = '/workspace/clear-ec-research/experiments/lane_cellroute_20261109/'
src = open(E + 'inference/test_budget.py').read()
blk = src[src.index("KEEP_NODET = "):src.index("    return True") + len("    return True")]
b = json.load(open(E + 'out/budget_fp32.json'))
for SEL in ('nodet', 'cdonly'):
    g = {'os': os, 're': re, 'SEL': SEL}; exec(blk, g); sel = g['selected']
    on = collections.defaultdict(float); off = collections.defaultdict(float)
    for f, v in b['files'].items():
        c = v.get('cast')
        if not c: continue
        k = '/'.join(f.split('/')[:3]) + (' e20' if f.endswith('epoch_20.pt') else '') + (' det' if '/detectors/' in f or '/frozen_heads/' in f else '')
        (on if sel('/workspace/clear-ec-research/' + f) else off)[k] += c[2] / 1e9
    print(SEL, 'cast %.2f GB, kept fp32 %.2f GB' % (sum(on.values()), sum(off.values())))
    for name, d in (('  cast', on), ('  keep', off)):
        for k, v in sorted(d.items(), key=lambda x: -x[1]): print(name, '%-80s %6.2f' % (k, v))
