"""Recipe A with a co-twin control term (2026-11-04). Fellow eyes share donor, session and device, so the difference of their errors cancels
everything but what is specific to one eye. Hybrid B explains little of that difference: within-pair r CD 0.70, CV 0.26, HEX 0.23, between-pair
0.91, 0.57, 0.56, and it predicts differences 0.29x (CV) and 0.24x (HEX) as wide as the labels'. Slope about 0.9, so what it predicts is real.
The frozen phase2_pooling_regression_20261019 runner with std pooling, except:
  order  each epoch's permutation is rearranged so rows 2k, 2k + 1 of the 16-row chunk are partners when a partner is in the fold's train set.
         Partners: new-format fellow eyes first, then old-format adjacent IDs (about half fellow by CD). Leftover rows pair with each other.
  loss   L(z) + lam L(difference): the same L1 + relative form on e_a - e_b for each adjacent pair in the 4-row microbatch, lam = 1.
         Leftover pairs are unrelated images; the term is then an ordinary differential loss.
No rows added or removed, validation unchanged. Read: terminal fold 0 against the matched seed-46 member, and within-pair r on validation."""
import re
from pathlib import Path

import numpy as np
import torch

import phase2_pooling_regression_20261019 as R

WRAPPER = Path(__file__).resolve()
LAM = 1.0
FROZEN_LOADER = R.loader
FROZEN_LOSS = R.b.loss_fn
FROZEN_SOURCES = R.source_files
FROZEN_CPU_TESTS = R.cpu_tests
FROZEN_SAVE = R.save
PARTNERS = None


def partners(ids):
    """Row -> partner row; each row in at most one pair per list. New-format fellow eyes win over old-format adjacent IDs."""
    old, new = {}, {}
    for n, i in enumerate(re.sub(r'\s+', ' ', str(v).strip()) for v in ids):
        m = re.fullmatch(r'(\d{4})-(\d{2})', i)
        if m:
            old[(int(m.group(2)), int(m.group(1)))] = n
            continue
        m = re.match(r'(\d{4})-(\d{4})[ -]*([O0][DS])', i)
        if m:
            new[(int(m.group(1)), int(m.group(2)), m.group(3).replace('0', 'O'))] = n
    out = {}
    for k, n in new.items():
        if k[2] == 'OD' and (k[0], k[1], 'OS') in new:
            c = new[(k[0], k[1], 'OS')]
            out.setdefault(n, []).append(c)
            out.setdefault(c, []).append(n)
    for k, n in old.items():
        if (k[0], k[1] + 1) in old:
            c = old[(k[0], k[1] + 1)]
            out.setdefault(n, []).append(c)
            out.setdefault(c, []).append(n)
    return out


def pair_order(order, table):
    """Same rows once each. Walks the permutation; a row's first unused partner in this train set follows it, leftovers pair among themselves."""
    in_set = set(int(i) for i in order)
    used, out, left = set(), [], []
    for i in map(int, order):
        if i in used:
            continue
        used.add(i)
        c = next((p for p in table.get(i, []) if p in in_set and p not in used), None)
        if c is None:
            left.append(i)
        else:
            used.add(c)
            out.append((i, c))
    pairs = [*out, *zip(left[::2], left[1::2])]
    rng = np.random.default_rng(int(order[0]))
    rows = [x for k in rng.permutation(len(pairs)) for x in pairs[k]]
    if len(left) % 2:
        rows.append(left[-1])
    return np.array(rows, dtype=order.dtype), len(out)


def loader(indices, targets, total, epoch=0, augment=False, batch_size=4):
    global PARTNERS
    if augment and batch_size == 16:
        if PARTNERS is None:
            PARTNERS = partners(np.load(R.b.META, allow_pickle=True)['ids'])
        indices, _ = pair_order(indices, PARTNERS)
    return FROZEN_LOADER(indices, targets, total, epoch, augment, batch_size)


def loss_fn(z, targets, mu, sigma):
    base = FROZEN_LOSS(z, targets, mu, sigma)
    n = len(z) // 2 * 2
    if n == 0:
        return base
    e = z.float() - (targets.log() - mu) / sigma
    d = e[:n:2] - e[1:n:2]
    # normalised by the rows in the microbatch, so microbatches sum to the same gradient as the whole chunk
    return base + LAM * (0.5 * d.abs().sum() + 2.5 * torch.expm1(d * sigma).abs().sum()) / (len(z) / 2 * z.shape[1])


def source_files():
    return FROZEN_SOURCES() + [WRAPPER]


def save(path, obj):
    if Path(path).name == 'manifest.json' and isinstance(obj, dict) and 'recipe' in obj:
        obj['recipe']['co_twin'] = dict(lam=LAM, order='partners adjacent in each chunk', partners='new-format fellow eyes, then old-format adjacent IDs')
    return FROZEN_SAVE(path, obj)


def cotwin_tests():
    ids = ['2019-0001 ODCN', '2019-0001 OSCN', '0005-19', '0006-19', '0009-19', '2019-0002 ODCN']
    t = partners(ids)
    assert t[0] == [1] and t[1] == [0] and t[2] == [3] and t[3] == [2] and 4 not in t and 5 not in t
    order = np.random.default_rng(0).permutation(6)
    rows, n = pair_order(order, t)
    assert sorted(rows.tolist()) == list(range(6)) and n == 2
    for a, c in ((0, 1), (2, 3)):
        assert abs(rows.tolist().index(a) - rows.tolist().index(c)) == 1 and min(rows.tolist().index(a), rows.tolist().index(c)) % 2 == 0
    mu, sigma = torch.tensor([7.9, -1.0, -0.6]), torch.tensor([0.3, 0.3, 0.2])
    y = torch.tensor([[2500., .3, .55], [2400., .35, .5], [2000., .4, .6], [2600., .3, .5]])
    z = (y.log() - mu) / sigma
    assert float(loss_fn(z, y, mu, sigma)) < 1e-5
    z2 = z + 0.3                                                    # a shared error costs the base term alone
    assert abs(float(loss_fn(z2, y, mu, sigma)) - float(FROZEN_LOSS(z2, y, mu, sigma))) < 1e-5
    z3 = z.clone()
    z3[0] += 0.3                                                    # an eye-specific error costs more
    assert float(loss_fn(z3, y, mu, sigma)) > float(FROZEN_LOSS(z3, y, mu, sigma)) + 1e-3
    real = partners(np.load(R.b.META, allow_pickle=True)['ids'])
    _, _, tr, va = R.arrays()
    rows, n = pair_order(np.random.default_rng(1).permutation(tr), real)
    assert np.array_equal(np.sort(rows), np.sort(tr))
    return dict(rows=int(len(tr)), partner_pairs=int(n), share_rows_paired=round(2 * n / len(tr), 3))


def cpu_tests():
    print({'cotwin_tests': cotwin_tests()}, flush=True)
    return FROZEN_CPU_TESTS()


R.loader = loader
R.b.loss_fn = loss_fn
R.source_files = source_files
R.save = save
R.cpu_tests = cpu_tests

if __name__ == '__main__':
    R.main()
