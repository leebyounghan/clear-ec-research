"""Why is the frozen std head on OOF encoder features so much worse than the members? Fold 0, recipe head, one head seed, std block only.
A: global z-score (the failed read). B: z-score per encoder. C: per-encoder z-score plus the member's own head on those features (no training). D: fit head on fold-0 encoder rows only (encoder 0 on its own validation rows, 2-fold within those rows) to separate encoder mismatch from the head fit itself."""
import sys, os, json
sys.argv = ['x']; sys.path.insert(0, 'source')
import importlib.util, numpy as np, torch
os.chdir('/workspace/clear-ec-research/experiments/lane_gpu2_20261106')
spec = importlib.util.spec_from_file_location('FR', 'source/frozen_read_20261106.py'); FR = importlib.util.module_from_spec(spec); spec.loader.exec_module(FR)
E = FR.E; y = FR.y; fold = FR.fold; good = FR.good; std = np.asarray(FR.block('std')).astype(np.float32)
def run(Xn, tr, va, seed=0, width=128):
    k = FR.fit(torch.from_numpy(Xn).cuda(), tr, va, seed, 'recipe', width); e = np.mean([E.ape(np.exp(q), y[va]) for q in k], 0); return E.official(e)
f = 0; tr = np.where((fold != f) & good)[0]; va = np.where(fold == f)[0]
m, s = std[tr].mean(0), std[tr].std(0) + 1e-6; print('A global z', run(np.clip((std - m) / s, -8, 8), tr, va), flush=True)
Z = std.copy()
for g in range(5):
    r = fold == g; Z[r] = (std[r] - std[r].mean(0)) / (std[r].std(0) + 1e-6)
print('B per-encoder z', run(np.clip(Z, -8, 8), tr, va), flush=True)
print('B2 raw no z', run(std, tr, va), flush=True)
rng = np.random.default_rng(0); own = rng.permutation(va); a, c = own[:1200], own[1200:]
print('D encoder 0 only, 1200 train rows, 591 val rows, raw', run(std, a, c), flush=True)
print('D ref: member on those 591 rows', E.official(E.ape(np.exp(np.load('frozen_s46/direct.npy', mmap_mode='r')[c].astype(np.float64)), y[c])), flush=True)
