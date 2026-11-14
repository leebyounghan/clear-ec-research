"""CV lane side read for the HEX lane: the same lane graph runs on the HEX column (graph level, folds 0+4): HEX from the new graph mean instead of the shipped graph."""
import sys
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/meta_ceiling_20261103/source')
import best_pipeline_20261106 as BP
E = BP.E; y = BP.y; fold = BP.fold; LO = BP.C.X / 'lane_cv_20261108/graph_runs'
for spec in sys.argv[1:]:
    Z = [np.load(LO / f'{r}.npz') for r in spec.split(',')]
    new = {f: (np.c_[BP.GRAPH[f][:, 0], BP.GRAPH[f][:, 1], np.mean([z[f'f{f}'][:, 2] for z in Z], 0)] if all(f'f{f}' in z for z in Z) else BP.GRAPH[f]) for f in range(5)}
    p = BP.pipe(new); rows = np.isin(fold, [0, 4]); a, a0 = E.ape(p, y), E.ape(BP.REF_GRAPH, y)
    print(spec, 'HEX folds 0+4: %.4f vs shipped graph %.4f  d %.4f' % (np.nanmean(a[rows, 2]), np.nanmean(a0[rows, 2]), np.nanmean(a[rows, 2]) - np.nanmean(a0[rows, 2])), flush=True)
