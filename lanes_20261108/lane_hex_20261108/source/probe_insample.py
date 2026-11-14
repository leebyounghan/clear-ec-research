import sys; sys.path.insert(0, '/workspace/clear-ec-research/experiments/lane_hex_20261108/source')
import hexlib as L, numpy as np
y, fold, N = L.y, L.fold, L.N; ok = y[:, 2] > 0
for f in (0, 4):
    tr = (fold != f) & ok; te = (fold == f) & ok
    e_in = np.mean(np.abs(L.HB[f][tr] - y[tr, 2]) / y[tr, 2]) * 100; e_te = np.mean(np.abs(L.HB[f][te] - y[te, 2]) / y[te, 2]) * 100
    oof = np.array([L.HB[fold[r]][r] for r in np.where(tr)[0]]); e_oof = np.mean(np.abs(oof - y[tr, 2]) / y[tr, 2]) * 100
    print(f'fold {f}: model-f HEX pre-stretch MAPE on its training rows {e_in:.3f}, same rows OOF {e_oof:.3f}, held-out fold {e_te:.3f}')
L.evaluate('control: ridge on log HB only (in-sample HB)', L.nested({f: np.zeros(N) for f in range(5)}))
