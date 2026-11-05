"""Side-channel probe 1 (2026-11-04): is there structure along the case ID order that the models cannot see?
For both ID formats, pairs at ID distance d within a year: correlation of log labels and of hybrid B's log residuals. Compared
with fellow eyes. Numbering gaps per year (is the held-out test set scattered among train numbers?). Size of the prize if a
neighbour's label were known: p' = B exp(beta r_neighbour), beta fit leave-one-fold-out, on the rows that have such a neighbour."""
import json, re, sys
import numpy as np
sys.path.insert(0, '/workspace/clear-ec-research/experiments/eval_harness_20261031/source')
import eval_oof_20261031 as E
K = list(E.K)
b = E.load_base(); y, B, fold = b['y'], b['B'], b['fold']; ids = [re.sub(r'\s+', ' ', i.strip()) for i in b['ids']]
ly = np.log(np.where(y > 0, y, np.nan)); r = ly - np.log(B)
old, new = {}, {}
for n, i in enumerate(ids):
    m = re.fullmatch(r'(\d{4})-(\d{2})', i)
    if m:
        old[(2000 + int(m.group(2)), int(m.group(1)))] = n; continue
    m = re.match(r'(\d{4})-(\d{4})[ -]*([O0][DS])', i)
    if m:
        new[(int(m.group(1)), int(m.group(2)), m.group(3).replace('0', 'O'))] = n
print('rows old', len(old), 'new', len(new), 'other', len(ids) - len(old) - len(new))
print('old-format years', {yr: sum(1 for k in old if k[0] == yr) for yr in sorted({k[0] for k in old})})
print('new-format years', {yr: sum(1 for k in new if k[0] == yr) for yr in sorted({k[0] for k in new})})
for yr in sorted({k[0] for k in old}):
    nums = sorted(k[1] for k in old if k[0] == yr); span = nums[-1] - nums[0] + 1
    print(f'  old {yr}: n {len(nums)} range {nums[0]}-{nums[-1]} present share {len(nums) / span:.3f}')
for yr in sorted({k[0] for k in new}):
    nums = sorted({k[1] for k in new if k[0] == yr}); span = nums[-1] - nums[0] + 1
    both = sum(1 for n in nums if (yr, n, 'OD') in new and (yr, n, 'OS') in new)
    print(f'  new {yr}: donors {len(nums)} both eyes {both} range {nums[0]}-{nums[-1]} present share {len(nums) / span:.3f}')
def pairs_old(d):
    return [(old[k], old[(k[0], k[1] + d)]) for k in old if (k[0], k[1] + d) in old]
def pairs_new_eye():
    return [(new[k], new[(k[0], k[1], 'OS')]) for k in new if k[2] == 'OD' and (k[0], k[1], 'OS') in new]
def pairs_new(d):
    return [(new[k], new[(k[0], k[1] + d, k[2])]) for k in new if (k[0], k[1] + d, k[2]) in new]
def cor(P):
    P = np.array(P); out = {}
    for j, k in enumerate(K):
        ok = np.isfinite(r[P[:, 0], j]) & np.isfinite(r[P[:, 1], j]); a, c = P[ok, 0], P[ok, 1]
        out[k] = (round(float(np.corrcoef(ly[a, j], ly[c, j])[0, 1]), 3), round(float(np.corrcoef(r[a, j], r[c, j])[0, 1]), 3))
    return dict(n=len(P), same_fold=round(float((fold[P[:, 0]] == fold[P[:, 1]]).mean()), 3), label_resid=out)
print('fellow eyes (new)', cor(pairs_new_eye()))
for d in (1, 2, 3, 5, 10, 50, 200):
    print(f'old d={d}', cor(pairs_old(d)))
for d in (1, 2, 3, 5, 10, 50):
    print(f'new same-side d={d}', cor(pairs_new(d)))
def prize(P, name):
    P = np.array(P + [(c, a) for a, c in P]); rows = np.unique(P[:, 0])
    p = B.copy(); betas = {}
    for f in range(5):
        for j, k in enumerate(K):
            ok = np.isfinite(r[P[:, 0], j]) & np.isfinite(r[P[:, 1], j]) & (fold[P[:, 0]] != f)
            x, t = r[P[ok, 1], j], r[P[ok, 0], j]
            be = float((x * t).sum() / (x * x).sum()); betas[f'{f}{k}'] = round(be, 3)
            te = np.isfinite(r[P[:, 1], j]) & (fold[P[:, 0]] == f) & (y[P[:, 0], j] > 0)
            # average over a row's neighbours
            s = np.zeros(len(B)); c = np.zeros(len(B)); np.add.at(s, P[te, 0], r[P[te, 1], j]); np.add.at(c, P[te, 0], 1)
            has = c > 0; p[has, j] = B[has, j] * np.exp(be * s[has] / c[has])
    a, c = E.official(E.ape(B[rows], y[rows])), E.official(E.ape(p[rows], y[rows]))
    print(name, 'rows', len(rows), 'B', {k: round(a[k], 3) for k in K + ['mean']}, '-> with neighbour', {k: round(c[k], 3) for k in K + ['mean']}, 'beta fold0', {k: betas[f'0{k}'] for k in K})
odd = [(old[k], old[(k[0], k[1] + 1)]) for k in old if (k[0], k[1] + 1) in old and k[1] % 2 == 1]
even = [(old[k], old[(k[0], k[1] + 1)]) for k in old if (k[0], k[1] + 1) in old and k[1] % 2 == 0]
print('old d=1, n odd', cor(odd)); print('old d=1, n even', cor(even))
for yr in sorted({k[0] for k in old}):
    o = [(old[k], old[(k[0], k[1] + 1)]) for k in old if k[0] == yr and (k[0], k[1] + 1) in old and k[1] % 2 == 1]
    e = [(old[k], old[(k[0], k[1] + 1)]) for k in old if k[0] == yr and (k[0], k[1] + 1) in old and k[1] % 2 == 0]
    print(yr, 'odd', len(o), cor(o)['label_resid']['CD'], 'even', len(e), cor(e)['label_resid']['CD'])
P = np.array(odd + even); C = np.abs(ly[P[:, 0], 0] - ly[P[:, 1], 0])
print('old d=1 |log CD diff| quantiles', np.round(np.nanquantile(C, [.1, .25, .5, .75, .9]), 3).tolist(), 'random pairs', np.round(np.nanquantile(np.abs(ly[:, 0] - np.random.default_rng(0).permutation(ly[:, 0])), [.1, .25, .5, .75, .9]), 3).tolist())
prize(pairs_new_eye(), 'PRIZE fellow eye (new)')
prize(odd, 'PRIZE old d=1 n odd'); prize(even, 'PRIZE old d=1 n even'); prize(odd + even, 'PRIZE old d=1 all')
# does a fellow in the training set already help the model? rows with an adjacent ID in another fold vs in the same fold or none
e = E.ape(B, y); ea = E.ape(b['A'], y)
adj = {}
for a_, c_ in odd + even:
    adj.setdefault(a_, []).append(c_); adj.setdefault(c_, []).append(a_)
oldrows = np.array(sorted(old.values())); other = np.array([any(fold[n] != fold[r_] for n in adj.get(r_, [])) for r_ in oldrows]); same = np.array([bool(adj.get(r_)) and all(fold[n] == fold[r_] for n in adj[r_]) for r_ in oldrows]); none = np.array([not adj.get(r_) for r_ in oldrows])
for nm, m in (('adjacent ID in another fold (seen in training)', other), ('adjacent ID only in the same fold (unseen)', same), ('no adjacent ID', none)):
    print(nm, int(m.sum()), {k: round(v, 3) for k, v in E.official(e[oldrows[m]]).items() if k != 'n'}, 'recipe A', round(E.official(ea[oldrows[m]])['mean'], 3))
print('new-format rows', {k: round(v, 3) for k, v in E.official(e[np.array(sorted(new.values()))]).items() if k != 'n'})
