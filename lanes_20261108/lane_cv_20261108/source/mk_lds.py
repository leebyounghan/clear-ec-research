p = 'source/family_cv_20261108.py'; s = open(p).read()
old = "def lane_loss(z, targets, mu, sigma):\n"
assert s.count(old) == 1
s = s.replace(old, '''LDS = {}   # CV lane 11-11: label-distribution smoothing (KG idea-imbalanced-regression-cv-tails)


def lds_init(cv_train):
    """Per-sample weight = 1 / Gaussian-KDE density of the training rows' log CV (Silverman bandwidth), mean 1 over the training rows,
    capped at 5 and renormalised once. Fixed a priori; nothing chosen on reported rows."""
    v = np.log(np.asarray(cv_train, np.float64)); h = 1.06 * v.std() * len(v) ** (-0.2); g = np.linspace(v.min() - 3 * h, v.max() + 3 * h, 1024)
    dens = np.zeros_like(g)
    for c in np.array_split(v, max(1, len(v) // 512)): dens += np.exp(-0.5 * ((g[:, None] - c[None, :]) / h) ** 2).sum(1)
    w = 1.0 / np.maximum(dens, 1e-12); wt = np.interp(v, g, w); w = w / wt.mean(); w = np.minimum(w, 5.0); wt = np.interp(v, g, w); w = w / wt.mean()
    LDS.update(g0=float(g[0]), dg=float(g[1] - g[0]), w=torch.tensor(w, dtype=torch.float32, device='cuda'), h=float(h), wmax=float(np.interp(v, g, w).max()))
    print(json.dumps(dict(lds_bandwidth=round(h, 4), weight_max=round(LDS['wmax'], 3), weight_p50=round(float(np.median(np.interp(v, g, w))), 3))), flush=True)


def lane_loss(z, targets, mu, sigma):
''')
old2 = "    if CD_LOSS == 'logl1':\n"
assert s.count(old2) == 1
s = s.replace(old2, '''    if CD_LOSS == 'lds':   # recipe loss with per-sample weights from the CV label density (LDS)
        i = ((targets[:, 1].float().log() - LDS['g0']) / LDS['dg']).round().long().clamp(0, len(LDS['w']) - 1); w = LDS['w'][i]
        return ((0.5 * d.abs() + 2.5 * torch.expm1(d * sigma).abs()) * w[:, None]).mean()
''' + old2)
old3 = "    ids, y, tr, va = arrays(a.fold); n = len(ids)\n    net = build_net(a, a.seed, ssl_path)\n"
assert s.count(old3) == 1
s = s.replace(old3, "    ids, y, tr, va = arrays(a.fold); n = len(ids)\n    if CD_LOSS == 'lds': lds_init(y[tr, 1])\n    net = build_net(a, a.seed, ssl_path)\n")
s = s.replace("TARGET_SD=1 (11-10) trains column 1 on the cell-area SD = CV x 1e6 / CD instead of CV.", "TARGET_SD=1 (11-10) trains column 1 on the cell-area SD = CV x 1e6 / CD instead of CV; CD_LOSS=lds (11-11) weights each sample by 1 / smoothed CV label density.", 1)
open(p, 'w').write(s); print('ok')
