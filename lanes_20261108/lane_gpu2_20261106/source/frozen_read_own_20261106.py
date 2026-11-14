"""F1 own, lane GPU 2 (2026-11-06), fallback in the gate: features from the fold's own encoder on all rows, head fit on its training rows, read on its validation rows. Uses frozen_read_20261106.py for the head and the read; only the feature folder per fold and the results folder change.
Usage: frozen_read_own_20261106.py <arm,arm,...|F2:kind,...> [--folds 0,4]"""
import importlib.util, json, os, sys
from pathlib import Path
L = Path('/workspace/clear-ec-research/experiments/lane_gpu2_20261106')
spec = importlib.util.spec_from_file_location('FR', L / 'source/frozen_read_20261106.py'); FR = importlib.util.module_from_spec(spec); spec.loader.exec_module(FR)
sys.path.insert(0, str(FR.X / 'meta_ceiling_20261103/source'))
FR.READS = FR.LANE / 'reads/own'; FR.READS.mkdir(parents=True, exist_ok=True)
folds = [int(v) for v in FR.opt('--folds', '0,4').split(',')]; seeds = [0, 1, 2]


def per_fold(name, feats, kind='recipe'):
    out = {}
    for f in folds:
        FR.OUT = FR.LANE / f'frozen_own_f{f}'; out.update(FR.read_arm(name, feats, [f], seeds, kind))
    return out


def ref():
    if all((FR.READS / f'std_f{f}.npz').exists() for f in folds): return FR.load_arm('std', folds)
    return per_fold('std', FR.ARMS['std'])


R0 = ref()
for arm in sys.argv[1].split(','):
    if arm.startswith('F2:'):
        kind = arm[3:]; a = per_fold(f'head_{kind}', ['std'], kind); name = f'head_{kind}'
    elif arm == 'std': continue
    else:
        a = per_fold(arm, FR.ARMS[arm]); name = arm
    o = FR.compare(name, a, R0, folds); (FR.READS / f'own_{"_".join(map(str, folds))}.jsonl').open('a').write(json.dumps(o) + '\n')
