"""HEX-specific weight between recipe A and Phase-1 in the base (B uses 0.5 for every metric). The base HEX becomes
exp(w log P1 + (1-w) log A); hex6 blend, graph + simulation corrections and stretch as in the best pipeline. w is read on all
folds and also chosen nested (inner choice on the four other folds)."""
import sys; sys.path.insert(0, '/workspace/clear-ec-research/experiments/lane_hex_20261108/source')
import hexlib as L, numpy as np
BP = L.BP; b = BP._b; A, P1 = b['A'], b['P1']; y, fold, N = L.y, L.fold, L.N; h = BP._have; allr = np.ones(N, bool)
def base_for(w):
    H2 = BP.H.copy(); bh = np.exp(w * np.log(P1[:, 2]) + (1 - w) * np.log(A[:, 2])); H2[:, 2] = bh
    H2[h, 2] = np.exp((1 - 0.28) * np.log(bh[h]) + 0.28 * (-0.2084 + 0.5989 * BP.f6[h])); return H2
W = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]; PP = {}
for w in W:
    PP[w] = BP.pipe(BP.BEST, base=base_for(w)); print('w', w, 'HEX all5 %.4f f04 %.4f' % (L.hexape(PP[w], allr), L.hexape(PP[w], L.S04)), [round(L.hexape(PP[w], fold == f), 3) for f in range(5)], flush=True)
# nested: choose w for fold f by the mean over the other folds (rows' own OOF values)
Pn = np.full_like(PP[0.5], np.nan); ch = []
for f in range(5):
    sc = {w: L.hexape(PP[w], fold != f) for w in W}; wb = min(sc, key=sc.get); ch.append(wb); Pn[fold == f] = PP[wb][fold == f]
print('nested choices', ch); d, ci = L.boot(Pn, allr); print('nested HEX all5 %.4f' % L.hexape(Pn, allr), 'delta %.4f' % d, ci, 'f04 %.4f' % L.hexape(Pn, L.S04))
