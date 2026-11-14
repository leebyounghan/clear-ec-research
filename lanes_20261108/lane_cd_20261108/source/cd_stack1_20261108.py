"""Stacking small CD gains on top of the family blend (folds 0, 4, cross-fit; settings for fold f chosen on the other screen fold only).
Base: P1/A2/F global geometric blend. Stacks: (a) per-source stretch after the blend; (b) detector density D blended on source B rows only
(one weight, chosen on the other fold), then stretch; (c) both."""
import numpy as np
from cd_common import *
import cd_pool_screen3_20261108 as P3
P1, A2, F, D, R1 = P3.P1, P3.A2, P3.F, P3.D, P3.R1
g3 = P3.g3
def run(stack_src_stretch, d_new):
    L = [np.log(P1), np.log(A2), np.log(F)]; out = np.full_like(P1, np.nan); info = {}
    for f, o in ((0, 4), (4, 0)):
        best = None
        for wt in g3:
            q = np.exp(sum(w * l for w, l in zip(wt, L)))
            for wd in (d_new if d_new else (0,)):
                qq = np.where(SRC_NEW, np.exp((1 - wd) * np.log(q) + wd * np.log(D)), q)
                if stack_src_stretch:
                    s = np.full_like(qq, np.nan)
                    for sv in (False, True):
                        m = SRC_NEW == sv; s[m] = apply1(qq[m], fit_stretch1(qq, (fold == o) & good & m))
                else:
                    s = apply1(qq, fit_stretch1(qq, (fold == o) & good))
                e = cd(s, fold == o)
                if best is None or e < best[0]: best = (e, wt, wd)
        _, wt, wd = best; info[f] = (wt, wd)
        q = np.exp(sum(w * l for w, l in zip(wt, L))); qq = np.where(SRC_NEW, np.exp((1 - wd) * np.log(q) + wd * np.log(D)), q)
        if stack_src_stretch:
            for sv in (False, True):
                m = SRC_NEW == sv; sel = (fold == f) & m; out[sel] = apply1(qq[sel], fit_stretch1(qq, (fold == o) & good & m))
        else:
            out[fold == f] = apply1(qq[fold == f], fit_stretch1(qq, (fold == o) & good))
    return out, info
base, i0 = run(False, None); rep('family blend (base)', base, R1, note=str(i0))
for nm, a, d in (('+ per-source stretch', True, None), ('+ D on source B', False, (0, 0.05, 0.1, 0.15, 0.2)), ('+ both', True, (0, 0.05, 0.1, 0.15, 0.2))):
    p, i = run(a, d); rep(nm + ' vs B', p, R1, note=str(i)); rep(nm + ' vs base', p, base)
