p = 'source/cv_deploy_fit_20261110.py'; s = open(p).read()
old = "    P = CA.load_params(); D = np.load(DEP / 'cv_deploy_insample_members.npz'); o, cz, names = oof()"
assert s.count(old) == 1
s = s.replace(old, "    P = CA.load_params(DEP); D = np.load(DEP / 'cv_deploy_insample_members.npz'); o, cz, names = oof()")
open(p, 'w').write(s); print('fixed')
