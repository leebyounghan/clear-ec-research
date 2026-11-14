p = 'source/cv_stack5_20261108.py'; s = open(p).read()
old = "    np.savez_compressed(X / f'lane_cv_20261108/out/{SAVE}.npz', P=P, names=np.array(names), w_all=np.array(W_ALL), **{f'w{f}': np.array(W[f]) for f in FOLDS}); print('saved', SAVE)"
assert s.count(old) == 1
new = ("    np.savez_compressed(X / f'lane_cv_20261108/out/{SAVE}.npz', P=P, names=np.array(names), w_all=np.array(W_ALL), **{f'w{f}': np.array(W[f]) for f in FOLDS}); print('saved', SAVE)\n"
       "    if os.environ.get('SAVE_COMPS'):  # component vectors per outer fold (CV column, all rows), for cv_apply_20261110.py check\n"
       "        np.savez_compressed(X / f'lane_cv_20261108/out/{SAVE}_comps.npz', names=np.array(names), **{f'c{k}_f{f}': comps[nm][f] for k, nm in enumerate(names) for f in FOLDS}); print('saved comps', SAVE)")
s = s.replace(old, new)
open(p, 'w').write(s); print('ok')
