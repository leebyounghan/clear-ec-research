"""Compare the no-stub deploy rerun (out/gpu_deploy_20b.json) with the earlier run whose cps5 was stubbed from OOF (out/gpu_deploy_20.json)."""
import json, numpy as np
E = '/workspace/clear-ec-research/experiments/lane_cellroute_20261109/out/'
a = json.load(open(E + 'gpu_deploy_20.json')); b = json.load(open(E + 'gpu_deploy_20b.json'))
print('old summary', json.dumps(a.get('summary', {}))[:600]); print('new summary', json.dumps(b.get('summary', {}))[:600])
ra = {r['frame']: r for r in a['rows']}; rb = b['rows']
assert all(r['frame'] in ra for r in rb)
for c in ('CD', 'CV', 'HEX'):
    x = np.array([ra[r['frame']][c] for r in rb]); y = np.array([r[c] for r in rb])
    print(c, 'old |deploy/OOF-1| median %.4f max %.4f -> new median %.4f max %.4f' % (np.median(x), x.max(), np.median(y), y.max()))
cd = np.array([r['cps5_deploy'] for r in rb]); co = np.array([r['cps5_oof'] for r in rb]); lr = np.log(cd / co)
print('cps5 deploy vs OOF: r %.3f, mean log ratio %+.4f, median |rel| %.4f, max |rel| %.4f' % (np.corrcoef(cd, co)[0, 1], lr.mean(), np.median(np.abs(cd / co - 1)), np.abs(cd / co - 1).max()))
hd = np.array([r['deploy'][2] for r in rb]); ho = np.array([r['oof'][2] for r in rb])
print('HEX deploy vs OOF: r %.4f, mean log ratio %+.4f' % (np.corrcoef(hd, ho)[0, 1], np.log(hd / ho).mean()))
print('seconds median', np.median([r['seconds'] for r in rb]), 'first', rb[0]['seconds'])
