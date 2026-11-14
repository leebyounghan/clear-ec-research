"""Shared-input identity of a tier run against the full run on their common frames (deploy lane, D-PKG-002/003): every per-image input the tier
shares with the full candidate must be identical (same members, same reads); DINO_logcv differs by design (one seed). usage: compare_shared.py OUT..."""
import json, sys
import numpy as np
L = '/workspace/clear-ec-research/experiments/lane_cellroute_20261109/out/'
full = {r['frame']: r for r in json.load(open(L + 'packaged_tier_full_3.json'))['rows']}
for name in sys.argv[1:]:
    rows = [r for r in json.load(open(L + name + '.json'))['rows'] if r['frame'] in full]; diff = {}
    for r in rows:
        for q, v in r['shared'].items():
            f = full[r['frame']]['shared'].get(q)
            if v is None or f is None or len(v) != len(f): continue
            a, b = np.asarray(v, float), np.asarray(f, float); m = np.isfinite(a) & np.isfinite(b)
            d = float(np.max(np.abs(a[m] - b[m]) / np.maximum(np.abs(b[m]), 1e-12))) if m.any() else 0.0
            diff[q] = max(diff.get(q, 0.0), d)
    print(name, 'common frames', len(rows), json.dumps({q: (0.0 if d == 0 else float('%.2e' % d)) for q, d in sorted(diff.items())}), flush=True)
