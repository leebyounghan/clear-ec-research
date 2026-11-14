p = 'source/family_cv_20261108.py'; s = open(p).read()
old = "    w = 1.0 / np.maximum(dens, 1e-12); wt = np.interp(v, g, w); w = w / wt.mean(); w = np.minimum(w, 5.0); wt = np.interp(v, g, w); w = w / wt.mean()\n"
assert s.count(old) == 1
new = """    w = 1.0 / np.maximum(dens, 1e-12)
    for _ in range(50):   # mean 1 over the training rows and max 5: alternate renormalise / cap until both hold
        w = w / np.interp(v, g, w).mean()
        if np.interp(v, g, w).max() <= 5.0 + 1e-6: break
        w = np.minimum(w, 5.0)
"""
s = s.replace(old, new)
s = s.replace("    capped at 5 and renormalised once. Fixed a priori; nothing chosen on reported rows.", "    capped at 5 (renormalise and cap alternately until both hold). Fixed a priori; nothing chosen on reported rows.")
open(p, 'w').write(s); print('ok')
