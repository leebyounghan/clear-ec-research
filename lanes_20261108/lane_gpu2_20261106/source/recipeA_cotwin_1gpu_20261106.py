"""Recipe A with the co-twin term on one GPU (lane GPU 2, idea G5, 2026-11-06). Same idea as recipeA_cotwin_20261104.py, which crashed on the runner's check that the rows seen equal the epoch's order, because it rearranged rows inside the loader.
Fix: the rearrangement now happens where the runner draws the order, through a numpy proxy in the runner's namespace, so the runner's order check, order hash and audit all see the paired order. Nothing else in the runner changes.
Rows 2k, 2k + 1 of each 16-row chunk are partners when a partner is in the fold's train set: fellow eyes first, then adjacent IDs of the old format. Leftover rows pair among themselves.
Loss: recipe objective + lam x the same L1 + relative form on (e_a - e_b) for each adjacent pair in the micro-batch; a shared error costs the base term only. lam = 1, as in the original design. Same rows, same flips, same schedule, validation unchanged."""
import os, re, sys
from pathlib import Path
import numpy as np
import torch
SRC = '/workspace/clear-ec-research/experiments/phase2_pooling_20261020/source'
sys.path.insert(0, SRC); sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase2_pooling_regression_20261019 as R
import recipeA_single_gpu_20261106 as S  # lane copy; patches R for one GPU

WRAPPER = Path(__file__).resolve(); LAM = float(os.environ.get('COTWIN_LAM', '1.0')); PARTNERS = None
FROZEN_LOSS = R.b.loss_fn; FROZEN_SOURCES = R.source_files; FROZEN_CPU_TESTS = R.cpu_tests; FROZEN_SAVE = R.save


def partners(ids):
    old, new = {}, {}
    for n, i in enumerate(re.sub(r'\s+', ' ', str(v).strip()) for v in ids):
        m = re.fullmatch(r'(\d{4})-(\d{2})', i)
        if m:
            old[(int(m.group(2)), int(m.group(1)))] = n; continue
        m = re.match(r'(\d{4})-(\d{4})[ -]*([O0][DS])', i)
        if m:
            new[(int(m.group(1)), int(m.group(2)), m.group(3).replace('0', 'O'))] = n
    out = {}
    for k, n in new.items():
        if k[2] == 'OD' and (k[0], k[1], 'OS') in new:
            c = new[(k[0], k[1], 'OS')]; out.setdefault(n, []).append(c); out.setdefault(c, []).append(n)
    for k, n in old.items():
        if (k[0], k[1] + 1) in old:
            c = old[(k[0], k[1] + 1)]; out.setdefault(n, []).append(c); out.setdefault(c, []).append(n)
    return out


def pair_order(order, table):
    in_set = set(int(i) for i in order); used, out, left = set(), [], []
    for i in map(int, order):
        if i in used: continue
        used.add(i); c = next((p for p in table.get(i, []) if p in in_set and p not in used), None)
        if c is None: left.append(i)
        else: used.add(c); out.append((i, c))
    pairs = [*out, *zip(left[::2], left[1::2])]; rng = np.random.default_rng(int(order[0])); rows = [x for k in rng.permutation(len(pairs)) for x in pairs[k]]
    if len(left) % 2: rows.append(left[-1])
    return np.array(rows, dtype=order.dtype), len(out)


class _Gen:
    def __init__(self, g): self.g = g
    def __getattr__(self, k): return getattr(self.g, k)
    def permutation(self, x):
        global PARTNERS
        p = self.g.permutation(x)
        if np.ndim(x) == 1 and len(p) > 1000:
            if PARTNERS is None: PARTNERS = partners(np.load(R.b.META, allow_pickle=True)['ids'])
            p, _ = pair_order(p, PARTNERS)
        return p


class _Random:
    def __getattr__(self, k): return getattr(np.random, k)
    def default_rng(self, *a, **k): return _Gen(np.random.default_rng(*a, **k))


class _Np:
    random = _Random()
    def __getattr__(self, k): return getattr(np, k)


def loss_fn(z, targets, mu, sigma):
    base = FROZEN_LOSS(z, targets, mu, sigma); n = len(z) // 2 * 2
    if n == 0: return base
    e = z.float() - (targets.log() - mu) / sigma; d = e[:n:2] - e[1:n:2]
    return base + LAM * (0.5 * d.abs().sum() + 2.5 * torch.expm1(d * sigma).abs().sum()) / (len(z) / 2 * z.shape[1])


def source_files(): return FROZEN_SOURCES() + [WRAPPER]


def save(path, obj):
    if Path(path).name == 'manifest.json' and isinstance(obj, dict) and 'recipe' in obj:
        obj['recipe']['co_twin'] = dict(lam=LAM, order='partners adjacent in each chunk', partners='new-format fellow eyes, then old-format adjacent IDs')
    return FROZEN_SAVE(path, obj)


def cotwin_tests():
    ids = ['2019-0001 ODCN', '2019-0001 OSCN', '0005-19', '0006-19', '0009-19', '2019-0002 ODCN']; t = partners(ids)
    assert t[0] == [1] and t[1] == [0] and t[2] == [3] and t[3] == [2] and 4 not in t and 5 not in t
    rows, n = pair_order(np.random.default_rng(0).permutation(6), t); assert sorted(rows.tolist()) == list(range(6)) and n == 2
    for a, c in ((0, 1), (2, 3)):
        assert abs(rows.tolist().index(a) - rows.tolist().index(c)) == 1 and min(rows.tolist().index(a), rows.tolist().index(c)) % 2 == 0
    mu, sigma = torch.tensor([7.9, -1.0, -0.6]), torch.tensor([0.3, 0.3, 0.2]); y = torch.tensor([[2500., .3, .55], [2400., .35, .5], [2000., .4, .6], [2600., .3, .5]]); z = (y.log() - mu) / sigma
    assert float(loss_fn(z, y, mu, sigma)) < 1e-5
    z2 = z + 0.3; assert abs(float(loss_fn(z2, y, mu, sigma)) - float(FROZEN_LOSS(z2, y, mu, sigma))) < 1e-5
    z3 = z.clone(); z3[0] += 0.3; assert float(loss_fn(z3, y, mu, sigma)) > float(FROZEN_LOSS(z3, y, mu, sigma)) + 1e-3
    real = partners(np.load(R.b.META, allow_pickle=True)['ids']); _, _, tr, va = R.arrays(); o1 = R.np.random.default_rng(1).permutation(tr); o2 = R.np.random.default_rng(1).permutation(tr)
    assert np.array_equal(o1, o2) and np.array_equal(np.sort(o1), np.sort(tr)), 'paired order deterministic, same rows'
    _, n = pair_order(np.random.default_rng(1).permutation(tr), real); share = 2 * n / len(tr); ok = sum(int(o1[k + 1]) in real.get(int(o1[k]), []) for k in range(0, len(o1) - 1, 2)); assert ok == n, 'partners sit in rows 2k, 2k + 1'
    return dict(rows=int(len(tr)), partner_pairs=int(n), share_rows_paired=round(share, 3), lam=LAM)


def cpu_tests():
    print({'cotwin_tests': cotwin_tests()}, flush=True); return FROZEN_CPU_TESTS()


S.MICRO = 4  # the runner's own micro-batch of 4: with the difference term, bf16 noise between micro-batch 8 and 4 sits just above the smoke threshold (0.0023 against 0.002)
R.np = _Np(); R.b.loss_fn = loss_fn; R.source_files = source_files; R.save = save; R.cpu_tests = cpu_tests
if __name__ == '__main__':
    R.main()
