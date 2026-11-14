"""Check masked_loss of bb_runner_full9000 (CPU): equals b.loss_fn when every label is positive; a row with HEX = 0 gets zero gradient on its
HEX output and nonzero gradient on CD and CV; finite."""
import sys, importlib.util, torch
sys.argv = ['x']
spec = importlib.util.spec_from_file_location('r9', 'experiments/lane_cd_20261108/source/bb_runner_full9000_20261113.py'); r9 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r9)
torch.manual_seed(0); mu = torch.tensor([7.9, -1.0, -0.6]); sigma = torch.tensor([0.12, 0.3, 0.2])
yy = torch.exp(mu + sigma * torch.randn(8, 3)); z = torch.randn(8, 3, requires_grad=True)
a = r9.masked_loss(z, yy, mu, sigma); bref = r9.b.loss_fn(z, yy, mu, sigma); print('all positive: masked', float(a), 'recipe', float(bref), 'diff', float(abs(a - bref)))
yy2 = yy.clone(); yy2[3, 2] = 0.0; z2 = z.detach().clone().requires_grad_(True); l2 = r9.masked_loss(z2, yy2, mu, sigma); l2.backward()
print('HEX-zero row: loss finite', bool(torch.isfinite(l2)), 'grad row 3 (CD, CV, HEX)', [round(float(v), 6) for v in z2.grad[3]], 'other rows HEX grad nonzero', bool((z2.grad[[0, 1, 2, 4], 2] != 0).all()))
z3 = z.detach().clone().requires_grad_(True); yy3 = yy.clone(); l3 = r9.masked_loss(z3, yy3, mu, sigma); l3.backward()
print('rows 0-2 gradients unchanged by masking row 3 HEX', bool(torch.allclose(z2.grad[[0, 1, 2, 4]], z3.grad[[0, 1, 2, 4]])))
