"""Exact duplicates show images filed under a mistyped ID: 11 pixel-identical pairs with different labels, B sides with one (median |CD residual| 0.019 vs 0.152). Copies are only the visible part: two files swapped, or a wrong file with no second copy, leave no duplicate.
Scan (2026-11-04): for ID pairs one digit apart or transposed (same format, year, eye), and fellow eyes, does exchanging the labels fit B better? Gain = sum over metrics of |r_a| + |r_c| before minus after. Excess over random same-year pairs estimates hidden swaps.
Also how far the top errors are explained. CPU."""
import json, re, sys
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/eval_harness_20261031/source')
import eval_oof_20261031 as E
b = E.load_base(); y, B, fold = b['y'], b['B'], b['fold']; ids = [re.sub(r'\s+', ' ', i.strip()) for i in b['ids']]; N = len(ids); K = list(E.K)
ly = np.log(np.where(y > 0, y, np.nan)); lB = np.log(B); key = {}
for n, i in enumerate(ids):
    m = re.fullmatch(r'(\d{4})-(\d{2})', i)
    if m: key[n] = ('old', m.group(2), m.group(1), ''); continue
    m = re.match(r'(\d{4})-(\d{4})[ -]*([O0][DS])', i)
    if m: key[n] = ('new', m.group(1), m.group(2), m.group(3).replace('0', 'O'))
def close(a, c):
    if len(a) != len(c): return False
    d = [k for k in range(len(a)) if a[k] != c[k]]
    return len(d) == 1 or (len(d) == 2 and a[d[0]] == c[d[1]] and a[d[1]] == c[d[0]] and d[1] == d[0] + 1)
groups = {}
for n, k in key.items(): groups.setdefault((k[0], k[1], k[3]), []).append(n)
pairs = []
for g in groups.values():
    for x in range(len(g)):
        for z in range(x + 1, len(g)):
            if close(key[g[x]][2], key[g[z]][2]): pairs.append((g[x], g[z]))
pairs = np.array(pairs); fellow = []
for n, k in key.items():
    if k[0] == 'new' and k[3] == 'OD':
        o = [m for m, q in key.items() if q == ('new', k[1], k[2], 'OS')] if False else None
byk = {v: n for n, v in key.items()}; fellow = np.array([(byk[k], byk[('new', k[1], k[2], 'OS')]) for k in byk if k[0] == 'new' and k[3] == 'OD' and ('new', k[1], k[2], 'OS') in byk])
def gain(P):
    a, c = P[:, 0], P[:, 1]; bef = np.abs(ly[a] - lB[a]) + np.abs(ly[c] - lB[c]); aft = np.abs(ly[c] - lB[a]) + np.abs(ly[a] - lB[c]); return np.nansum(bef - aft, 1), np.nansum(aft, 1), np.nansum(bef, 1)
rng = np.random.default_rng(0); null = []
for (fm, yr, eye), g in groups.items():
    n = sum(1 for p, q in pairs if key[p][:2] == (fm, yr) and key[p][3] == eye) if False else None
ix = {}; 
for p in pairs: ix.setdefault((key[p[0]][0], key[p[0]][1], key[p[0]][3]), []).append(p)
for k, v in ix.items():
    g = np.array(groups[k])
    for _ in range(20): null.append(np.c_[rng.choice(g, len(v)), rng.choice(g, len(v))])
null = np.concatenate(null); null = null[null[:, 0] != null[:, 1]]
G, Aft, Bef = gain(pairs); G0, A0, B0 = gain(null); GF, AF, BF = gain(fellow); print('pairs near IDs', len(pairs), 'fellow', len(fellow), 'null', len(null))
rep = {}
for t in (0.2, 0.4, 0.6, 0.8, 1.0):
    rep[t] = dict(near=int((G > t).sum()), near_expected_null=round(float((G0 > t).mean() * len(pairs)), 1), fellow=int((GF > t).sum()))
print('gain above t', json.dumps(rep))
print('fellow: mean before and after', round(float(BF.mean()), 3), round(float(AF.mean()), 3), 'near: before', round(float(Bef.mean()), 3), 'after', round(float(Aft.mean()), 3), 'null: before', round(float(B0.mean()), 3), 'after', round(float(A0.mean()), 3))
top = np.argsort(-G)[:14]; print('top near-ID candidates', json.dumps([dict(a=ids[pairs[k, 0]], c=ids[pairs[k, 1]], gain=round(float(G[k]), 2), before=round(float(Bef[k]), 2), after=round(float(Aft[k]), 2), ya=y[pairs[k, 0]].round(2).tolist(), yc=y[pairs[k, 1]].round(2).tolist(), Ba=B[pairs[k, 0]].round(2).tolist(), Bc=B[pairs[k, 1]].round(2).tolist()) for k in top]))
