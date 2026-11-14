p = '/workspace/clear-ec-research/experiments/lane_cv_20261108/source/graph_train_cv_20261108.py'; s = open(p).read()
# 1) full-data mode: --folds full trains on all good rows and saves the weights of the last 5 epochs per seed (the OOF read averages those 5 epochs)
old_run = "        if ep >= epochs - 5:\n            snaps += predict(net, allrows); ns += 1\n"
new_run = "        if ep >= epochs - 5:\n            snaps += predict(net, allrows); ns += 1\n            if f < 0: SAVE_DIR.mkdir(parents=True, exist_ok=True); torch.save(dict(model=net.state_dict(), d=A_.d, layers=A_.layers, epoch=ep + 1, seed=seed, graphs=A_.graphs, no_ctx=A_.no_ctx), SAVE_DIR / f'{A_.name}_s{seed}_e{ep + 1}.pt')\n"
assert old_run in s; s = s.replace(old_run, new_run, 1)
old = "FOLDS = [int(c) for c in A_.folds]; pred = {f: np.mean([run(f, seed=q) for q in range(A_.seeds)], 0) for f in FOLDS}"
new = ("SAVE_DIR = C.X / 'lane_cv_20261108/graph_models'\n"
       "if A_.folds == 'full':\n"
       "    full = np.mean([run(-1, seed=q) for q in range(A_.seeds)], 0); np.savez_compressed(SAVE_DIR / f'{A_.name}_fullfit_pred.npz', pred=full.astype(np.float32)); print('full-data models saved in', SAVE_DIR, flush=True); sys.exit(0)\n"
       "FOLDS = [int(c) for c in A_.folds]; pred = {f: np.mean([run(f, seed=q) for q in range(A_.seeds)], 0) for f in FOLDS}")
assert old in s; s = s.replace(old, new, 1)
s = s.replace('"""CV lane copy (2026-11-08): --folds limits', '"""CV lane copy (2026-11-08): --folds full trains on all good rows and saves the last-5-epoch weights per seed to graph_models/ (deploy). --folds limits', 1)
open(p, 'w').write(s); print('patched')
