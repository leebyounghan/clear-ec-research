p = 'experiments/lane_cd_20261108/source/bb_runner_full9000_20261113.py'; s = open(p).read()
def rep(a, b_):
    global s
    assert s.count(a) == 1, (s.count(a), a[:70]); s = s.replace(a, b_)
rep("""        tr = np.where(np.isfinite(y).all(1) & (y > 0).all(1))[0].astype(np.int64)
        va = np.sort(np.random.default_rng(12013).choice(tr, 256, replace=False))""",
"""        pos = np.where(np.isfinite(y).all(1) & (y > 0).all(1))[0].astype(np.int64)
        tr = np.where(np.isfinite(y).all(1) & (y[:, :2] > 0).all(1) & (y[:, 2] >= 0))[0].astype(np.int64)   # full9000b: HEX-zero rows train on CD and CV only
        va = np.sort(np.random.default_rng(12013).choice(pos, 256, replace=False))""")
rep("""    logy = np.log(y[tr].astype(np.float64)); mu = torch.tensor(logy.mean(0), device='cuda', dtype=torch.float32); sigma = torch.tensor(logy.std(0), device='cuda', dtype=torch.float32)""",
"""    _yt = y[tr].astype(np.float64); logy = np.where(_yt > 0, np.log(np.where(_yt > 0, _yt, 1.0)), np.nan)   # full9000b: per column on positive labels
    mu = torch.tensor(np.nanmean(logy, 0), device='cuda', dtype=torch.float32); sigma = torch.tensor(np.nanstd(logy, 0), device='cuda', dtype=torch.float32)""")
rep("""def arrays(fold):""", '''def masked_loss(z, yy, mu, sigma, TW=None):
    """recipe loss 0.5|d| + 2.5|expm1(d sigma)| per metric; a metric whose label is not positive gets weight 0 on that row; per-row
    normalisation by the full weight sum (3 for the recipe), so rows with three positive labels weigh exactly as in b.loss_fn"""
    m = (yy > 0).float(); zt = (torch.log(torch.where(yy > 0, yy, torch.ones_like(yy))) - mu) / sigma; d = z.float() - zt
    w = (TW if TW is not None else torch.ones(3, device=z.device)) * m
    return ((0.5 * d.abs() + 2.5 * torch.expm1(d * sigma).abs()) * w).sum(1).mean() / (TW.sum() if TW is not None else 3.0)


def arrays(fold):''')
rep("""                        loss = b.loss_fn(z[:, :3], yy, mu, sigma) if TW is None else ((0.5 * (z[:, :3].float() - (yy.log() - mu) / sigma).abs() + 2.5 * torch.expm1((z[:, :3].float() - (yy.log() - mu) / sigma) * sigma).abs()) * TW).sum(1).mean() / TW.sum()""",
    """                        loss = masked_loss(z[:, :3], yy, mu, sigma, TW)""")
rep("""rel = torch.expm1((z[:, :3].float() - (yy.log() - mu) / sigma) * sigma).abs()""",
    """rel = torch.expm1((z[:, :3].float() - (torch.log(torch.where(yy > 0, yy, torch.ones_like(yy))) - mu) / sigma) * sigma).abs() * (yy > 0)""")
rep("""full-data builds (fold -1, 8,998 rows) only""", """full-data builds (fold -1, all 9,000 rows) only""")
rep("# full9000 (11-13 23:45)", "# full9000b (11-13 23:55, PI): all 9,000 rows train; the HEX term is masked for the two HEX-zero rows (masked_loss); mu / sigma on positive labels. full9000 (11-13 23:45)")
open(p, 'w').write(s); print('patched')
