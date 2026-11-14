p = 'source/cv_stack5_20261108.py'; s = open(p).read()
old = "names = list(comps); rows = np.isin(fold, FOLDS); GRID = np.round(np.arange(0, 1.0001, 0.125), 3)\n"
assert s.count(old) == 1
new = ("if os.environ.get('STACK_DUMP'):   # 11-11: save the component vectors (CV column, per fold model) for the from-scratch search and stop\n"
       "    np.savez_compressed(X / f\"lane_cv_20261108/out/{os.environ['STACK_DUMP']}.npz\", names=np.array(list(comps)), **{f'c{k}_f{f}': comps[nm][f] for k, nm in enumerate(comps) for f in FOLDS}); print('dumped', len(comps), flush=True); sys.exit(0)\n" + old)
s = s.replace(old, new); open(p, 'w').write(s); print('ok')
