p = 'source/graph_order_cv_20261110.py'; s = open(p).read()
old = "full = np.mean([run(-1, seed=q) for q in range(A_.seeds)], 0)"
assert s.count(old) == 1
s = s.replace(old, "full = np.mean([run(-1, seed=q) for q in range(A_.seed0, A_.seed0 + A_.seeds)], 0)")
s = s.replace('--seed0 K runs seeds K..K+seeds-1 (to split seeds over processes).', '--seed0 K runs seeds K..K+seeds-1 (to split seeds over processes; fixed 2026-11-10 10:00 UTC for --folds full, which before ran seed 0 whatever --seed0 was).', 1)
open(p, 'w').write(s); print('patched')
