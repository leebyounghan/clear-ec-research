import os, numpy as np
os.environ.setdefault('SLOT', 'V'); os.environ.setdefault('EPOCHS', '4-5')
src = open('cd_interim_slot_20261114.py').read(); src = src[:src.index('# repeated grouped 10-fold')]
src = src.replace("out[f'fold{f}'] = o", "out[f'fold{f}'] = o; break")
g = {'__name__': 'diag'}; exec(compile(src, 'interim_head', 'exec'), g)
F04, ok, fold, ARM, PL, members, wsc, fit, P1raw, yc = [g[k] for k in ('F04', 'ok', 'fold', 'ARM', 'PL', 'members', 'wsc', 'fit', 'P1raw', 'yc')]
tr = F04 & ok
print('NaN in ARM CD on F04 rows:', int(np.isnan(ARM[F04, 0]).sum()), 'of', int(F04.sum()))
sPr = np.zeros(len(yc)); [sPr.__setitem__(fold == f, wsc(P1raw, fold != f)) for f in range(5)]
Ar, Vr = members(None); sAv = np.zeros(len(yc)); [sAv.__setitem__(fold == f, wsc(Ar, fold != f)) for f in range(5)]
print('NaN in sAv on F04:', int(np.isnan(sAv[F04]).sum()), 'good&F04:', int((g['good'] & F04).sum()))
for nm, ex in (('arm', ARM[:, 0]), ('plain', PL[:, 0])):
    P = fit(tr, sPr, sAv, Ar, Vr, ex, coarse=True); print(nm, 'extra fit w', P['w'], 'wmk', P['wmk'])
