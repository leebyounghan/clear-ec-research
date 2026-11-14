"""Exploratory, lane GPU 2 (2026-11-06): the saved F1b corrections applied to HEX only. The metric was chosen after seeing outer folds, so this cannot pass the correction rule.
Usage: corr_hexonly_20261106.py"""
import json, os, sys
from pathlib import Path
os.chdir('/workspace/clear-ec-research')
import numpy as np
X = Path('experiments'); sys.path.insert(0, str(X / 'meta_ceiling_20261103/source'))
import best_pipeline_20261106 as BP
D = X / 'lane_gpu2_20261106/reads/v2/corr'
for p in sorted(D.glob('*.npz')):
    c = BP.load(p); m = {f: c[f] * np.array([0., 0., 1.]) for f in range(5)}; o = BP.report(p.stem + '_hex', BP.pipe(BP.add(BP.BEST, m)), BP.REF_BEST)
    (D / 'hexonly.jsonl').open('a').write(json.dumps(o) + '\n')
print('hexonly done', flush=True)
