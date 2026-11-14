import sys, numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/lane_hex_20261108/source')
import hexlib as L
LANE = '/workspace/clear-ec-research/experiments/lane_hex_20261108'
pos = {int(f): r for r, f in enumerate(L.idx)}
ch = np.load(f'{LANE}/preds/chain_main.npz'); fh = ch['final_hex']
yD, yC, yH = L.y[:, 0], L.y[:, 1], L.y[:, 2]
rz = np.load(f'{LANE}/feats/ridge_k3.npz'); RF = rz['F']
res = np.log(yH) - np.log(fh)
for p in sys.argv[1:]:
    z = np.load(f'{LANE}/{p}'); F = z['F'].astype(float); nm = [str(q) for q in z['names']]; fr = z['frames']
    rows = np.array([pos[int(f)] for f in fr])
    ok = np.isfinite(F).all(1) & ((F[:, nm.index('n_int')] >= 30) if 'n_int' in nm else True)
    print(p, 'usable', ok.sum(), 'of', len(fr), 'median n_int', np.nanmedian(F[:, 3]), 'median cov %.3f' % np.nanmedian(F[:, 6]))
    r = rows[ok]
    for j, n in enumerate(nm):
        v = F[ok, j]
        print('  %-9s r(yHEX) %+.3f r(yCV) %+.3f r(yCD) %+.3f r(res) %+.3f' % (n, np.corrcoef(v, yH[r])[0, 1], np.corrcoef(v, yC[r])[0, 1], np.corrcoef(v, yD[r])[0, 1], np.corrcoef(v, res[r])[0, 1]))
    v = RF[fr[ok], 0]; v = np.where(np.isfinite(v), v, np.nanmedian(v))
    print('  voronoi h6_s15 same rows r(yHEX) %+.3f r(res) %+.3f' % (np.corrcoef(v, yH[r])[0, 1], np.corrcoef(v, res[r])[0, 1]))
