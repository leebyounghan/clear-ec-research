p = '/workspace/clear-ec-research/experiments/lane_cv_20261108/source/family_cv_20261108.py'; s = open(p).read()
old = "    if CD_LOSS == 'logl1':\n        return (d * sigma).abs().mean() * 5.\n"
new = old + """    if CD_LOSS == 'order':   # CV lane 11-09: recipe loss + 1.0 x noise-weighted pairwise order loss on CV (KG lee2026stochorder; s = 0.065 log units)
        base = b.loss_fn(z, targets, mu, sigma)
        if len(z) < 2: return base
        s1 = sigma[1].float(); dp = (z[:, 1].float()[:, None] - z[:, 1].float()[None, :]) * s1; dt = (zt[:, 1][:, None] - zt[:, 1][None, :]) * s1
        p = 0.5 * (1 + torch.erf(dt / (2 * 0.065))); m = ~torch.eye(len(z), dtype=torch.bool, device=z.device)
        return base + torch.nn.functional.binary_cross_entropy_with_logits(dp[m] / (math.sqrt(2) * 0.065), p[m])
"""
assert old in s and "CD_LOSS == 'order'" not in s; s = s.replace(old, new, 1); open(p, 'w').write(s); print('patched')
