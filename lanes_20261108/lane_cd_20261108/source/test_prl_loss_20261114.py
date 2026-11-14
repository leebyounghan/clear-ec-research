"""CPU check of prl_loss (bb_runner_prl_20261114): k = 0 equals b.loss_fn; k = 1 on a batch of 16 gives zero gradient to the trimmed term of each
metric and the other terms' gradients equal the untrimmed ones up to the normalisation factor 48 / 45."""
import sys, importlib.util, torch
sys.argv = ['x']
spec = importlib.util.spec_from_file_location('rp', 'experiments/lane_cd_20261108/source/bb_runner_prl_20261114.py'); rp = importlib.util.module_from_spec(spec); spec.loader.exec_module(rp)
torch.manual_seed(0); mu = torch.tensor([7.9, -1.0, -0.6]); sigma = torch.tensor([0.12, 0.3, 0.2]); yy = torch.exp(mu + sigma * torch.randn(16, 3))
z0 = torch.randn(16, 3); a = rp.prl_loss(z0, yy, mu, sigma, 0); print('k=0 vs recipe diff', float(abs(a - rp.b.loss_fn(z0, yy, mu, sigma))))
z1 = z0.clone().requires_grad_(True); rp.prl_loss(z1, yy, mu, sigma, 0).backward(); g0 = z1.grad.clone()
z2 = z0.clone().requires_grad_(True); rp.prl_loss(z2, yy, mu, sigma, 1).backward(); g1 = z2.grad.clone()
d = z0 - (yy.log() - mu) / sigma; top = torch.argmax(0.5 + 2.5 * sigma * torch.exp(d * sigma), 0)
zero_ok = all(float(g1[top[j], j]) == 0.0 for j in range(3)); m = torch.ones_like(g1, dtype=torch.bool); m[top, torch.arange(3)] = False
print('trimmed rows per metric', top.tolist(), 'their grad zero', zero_ok, 'others equal x48/45', bool(torch.allclose(g1[m], g0[m] * 48 / 45, rtol=1e-5, atol=1e-8)))
