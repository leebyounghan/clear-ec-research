src = 'experiments/lane_cd_20261108/source/bb_runner_sweep3_20261113.py'; dst = 'experiments/lane_cd_20261108/source/bb_runner_prl_20261114.py'
s = open(src).read()
def rep(a, b_):
    global s
    assert s.count(a) == 1, (s.count(a), a[:70]); s = s.replace(a, b_)
rep("""def arrays(fold):""", '''def prl_loss(z, yy, mu, sigma, k):
    """recipe loss 0.5|d| + 2.5|expm1(d sigma)| averaged over the kept terms; with k > 0, per metric the k rows of the micro-batch with the largest
    |d term / d z| = 0.5 + 2.5 sigma exp(d sigma) get weight 0 (PRL-style trimming, Liu et al. 2021); k = 0 is exactly b.loss_fn"""
    zt = (yy.log() - mu) / sigma; d = z.float() - zt; per = 0.5 * d.abs() + 2.5 * torch.expm1(d * sigma).abs()
    if k <= 0 or per.shape[0] <= k:
        return per.mean()
    with torch.no_grad():
        g = 0.5 + 2.5 * sigma * torch.exp(d * sigma); keep = torch.ones_like(per); keep.scatter_(0, torch.topk(g, k, dim=0).indices, 0.0)
    return (per * keep).sum() / keep.sum()


def arrays(fold):''')
rep("""p.add_argument("--target-w", default="")""", """p.add_argument("--target-w", default=""); p.add_argument("--prl", type=int, default=0)""")
rep("""                        loss = b.loss_fn(z[:, :3], yy, mu, sigma) if TW is None else""", """                        loss = prl_loss(z[:, :3], yy, mu, sigma, a.prl if epoch >= 1 else 0) if a.prl else b.loss_fn(z[:, :3], yy, mu, sigma) if TW is None else""")
rep("# sweep3 (11-13 23:35)", "# prl (11-14 00:40): --prl K drops, per metric and micro-batch, the K rows with the largest loss gradient w.r.t. the output, from epoch 2 on (prl_loss). sweep3 (11-13 23:35)")
open(dst, 'w').write(s); print('written', dst)
