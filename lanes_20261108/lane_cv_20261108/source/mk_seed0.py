p = 'source/graph_order_cv_20261110.py'; s = open(p).read()
s = s.replace("ap.add_argument('--order', type=float, default=0.0);", "ap.add_argument('--order', type=float, default=0.0); ap.add_argument('--seed0', type=int, default=0);", 1)
old = "pred = {f: np.mean([run(f, seed=q) for q in range(A_.seeds)], 0) for f in FOLDS}"
assert s.count(old) == 1
s = s.replace(old, "pred = {f: np.mean([run(f, seed=q) for q in range(A_.seed0, A_.seed0 + A_.seeds)], 0) for f in FOLDS}")
s = s.replace('GRAPH_DEV=cpu runs on CPU.', 'GRAPH_DEV=cpu runs on CPU; --seed0 K runs seeds K..K+seeds-1 (to split seeds over processes).', 1)
open(p, 'w').write(s); print('ok')
