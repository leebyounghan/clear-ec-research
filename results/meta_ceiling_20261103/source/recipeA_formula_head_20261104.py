"""Recipe A with a head that computes the label formulas from local maps (2026-11-04).

The frozen phase2_pooling_regression_20261019 runner, except that std pooling and its MLP are replaced by a structured head:
tokens (124 x 164 grid, 8 px patches) -> two small conv layers with a ~2-cell receptive field -> per-token maps of mean cell area
A(x), relative within-neighbourhood area variance v(x), share of six-sided cells s(x) and a ROI score. The ROI score gives a
distribution over box centres, and the weight map w(x) is that distribution convolved with the technician box (408 x 538 px, 51 x
67 tokens). The same w feeds all three outputs:
  m = sum w A;  CD = c / m;  CV = sqrt(sum w (A^2 v + (A - m)^2)) / m;  HEX = sum w s
which mirrors how the labels are made: a statistic of cells inside one box. Initialised to the label means with a uniform w, as
recipe A's head starts near zero. Head parameters train at the head learning rate, the encoder keeps 1e-5.
Why: bridge capacity tests (encoder LR x5, per-token MLP before std pooling) did not rule out a head with this structure. Nothing
forces the maps to mean literally cells or the box to be the true ROI; labels are the only supervision here.
Read: fold 0 terminal epoch against the matched seed-46 member, and as an added ensemble member, protocol in formula_head/.
"""
import math
from pathlib import Path

import torch
import torch.nn.functional as F

import phase2_pooling_20261019 as pooling
import phase2_pooling_regression_20261019 as R

WRAPPER = Path(__file__).resolve()
GRID = (124, 164)
BOX = (51, 67)
INIT = dict(log_c=7.92, log_v=-2.0, six_logit=math.log(0.54 / 0.46))
FROZEN_BUILD = R.build_cpu
FROZEN_SETUP = R.setup
FROZEN_SOURCES = R.source_files
FROZEN_SAVE = R.save


class FormulaHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Conv2d(768, 128, 1)
        self.ctx = torch.nn.Sequential(torch.nn.Conv2d(128, 128, 3, padding=1), torch.nn.GELU(),
                                       torch.nn.Conv2d(128, 128, 3, padding=2, dilation=2), torch.nn.GELU())
        self.area, self.var, self.six, self.roi = (torch.nn.Conv2d(128, 1, 1) for _ in range(4))
        for out, bias in ((self.area, 0.0), (self.var, INIT['log_v']), (self.six, INIT['six_logit']), (self.roi, 0.0)):
            torch.nn.init.normal_(out.weight, std=1e-3)
            torch.nn.init.constant_(out.bias, bias)
        self.log_c = torch.nn.Parameter(torch.tensor(INIT['log_c']))
        self.offset = torch.nn.Parameter(torch.zeros(3))
        self.slope = torch.nn.Parameter(torch.ones(3))
        self.register_buffer('mu', torch.tensor([INIT['log_c'], INIT['log_v'] / 2, math.log(0.54)]))
        self.register_buffer('sigma', torch.ones(3))

    def maps(self, tokens):
        B, N, C = tokens.shape
        assert N == GRID[0] * GRID[1], N
        h = self.ctx(self.proj(tokens.float().transpose(1, 2).reshape(B, C, *GRID)))
        return (self.area(h).clamp(-3, 3).exp(), self.var(h).clamp(-8, 2).exp(), torch.sigmoid(self.six(h)), self.roi(h))

    def weights(self, roi):
        pi = F.softmax(roi.flatten(1), 1).view_as(roi)
        w = F.avg_pool2d(pi, BOX, stride=1, padding=(BOX[0] // 2, BOX[1] // 2))
        return w / w.sum((2, 3), keepdim=True)

    def forward(self, tokens):
        with torch.autocast('cuda', enabled=False):
            A, v, s, roi = self.maps(tokens)
            w = self.weights(roi)
            total = lambda f: (w * f).sum((1, 2, 3))
            m = total(A)
            var = total(A ** 2 * v + (A - m[:, None, None, None]) ** 2)
            log_y = torch.stack([self.log_c - m.log(), 0.5 * (var + 1e-8).log() - m.log(), total(s).clamp_min(1e-4).log()], 1)
            return self.slope * (log_y - self.mu) / self.sigma + self.offset


def forward_tokens(self, x):
    return self.original.forward_features(x)['x_norm_patchtokens']


def build_cpu(seed):
    net, details = FROZEN_BUILD(seed)
    torch.manual_seed(seed)
    details['head'] = 'formula head'
    return torch.nn.Sequential(net[0], FormulaHead()), details


def setup(m):
    net, opt, mu, sigma, details = FROZEN_SETUP(m)
    head = net[1]
    head.mu.copy_(torch.tensor(m['log_mean'], dtype=head.mu.dtype, device=head.mu.device))
    head.sigma.copy_(torch.tensor(m['log_std'], dtype=head.sigma.dtype, device=head.sigma.device))
    with torch.no_grad():                                       # start each output at its train log mean: z = 0
        head.log_c.copy_(head.mu[0])
        head.var.bias.fill_(2 * float(head.mu[1]))
        p = float(head.mu[2].exp())
        head.six.bias.fill_(math.log(p / (1 - p)))
    return net, opt, mu, sigma, details


def save(path, obj):
    if Path(path).name == 'manifest.json' and isinstance(obj, dict) and 'recipe' in obj:
        obj['recipe']['head'] = 'formula head: tokens -> conv maps A, v, s, ROI score; box-convolved weights; CD = c/m, CV from weighted area variance, HEX = weighted s'
        obj['recipe']['pooling_note'] = 'pooling argument unused: the encoder returns tokens'
    return FROZEN_SAVE(path, obj)


def source_files():
    return FROZEN_SOURCES() + [WRAPPER]


def head_tests():
    t = torch.randn(2, GRID[0] * GRID[1], 768, dtype=torch.float32)
    head = FormulaHead()
    z = head(t)
    assert z.shape == (2, 3) and torch.isfinite(z).all()
    A, v, s, roi = head.maps(t)
    w = head.weights(roi)
    torch.testing.assert_close(w.sum((2, 3)), torch.ones_like(w.sum((2, 3))), rtol=1e-5, atol=1e-5)
    y = z * head.sigma + head.mu
    assert abs(float(y[:, 0].mean() - INIT['log_c'])) < 0.2 and abs(float(y[:, 1].mean() - INIT['log_v'] / 2)) < 0.2 and abs(float(y[:, 2].mean() - head.mu[2])) < 0.2
    # known values: constant maps give CD = c / A, CV = sqrt(v), HEX = s for any weights
    A0, v0, s0 = torch.full_like(A, 2.0), torch.full_like(v, 0.09), torch.full_like(s, 0.5)
    m = (w * A0).sum((1, 2, 3)); var = (w * (A0 ** 2 * v0 + (A0 - m[:, None, None, None]) ** 2)).sum((1, 2, 3))
    torch.testing.assert_close(var.sqrt() / m, torch.full_like(m, 0.3), rtol=1e-4, atol=1e-4)
    # spatial spread in A raises CV: two regions of different area, same within-region variance
    A1 = A0.clone(); A1[:, :, :, :GRID[1] // 2] = 1.0
    w0 = torch.full_like(w, 1 / (GRID[0] * GRID[1]))
    m1 = (w0 * A1).sum((1, 2, 3)); var1 = (w0 * (A1 ** 2 * v0 + (A1 - m1[:, None, None, None]) ** 2)).sum((1, 2, 3))
    assert (var1.sqrt() / m1 > 0.3).all()
    z.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    assert head.area.weight.grad.abs().sum() > 0 and head.six.weight.grad.abs().sum() > 0 and head.roi.weight.grad.abs().sum() > 0
    return dict(params=sum(p.numel() for p in head.parameters()), init_log_y=[round(float(v), 3) for v in y.mean(0)], known_values=True, gradients=True)


def cpu_tests():
    m = R.verify()
    assert not (R.ROOT / 'cpu_tests.json').exists(), 'CPU test evidence already exists'
    pooling.cpu_contract_tests()
    R.b.tests()
    rep = head_tests()
    net, _ = build_cpu(R.ARGS.seed)
    assert R.prior.state_sha(net[1:].state_dict()) == m['expected_head_sha']
    w0 = net[1].proj.weight.detach().clone()
    net, _ = build_cpu(R.ARGS.seed + 1)
    assert R.prior.state_sha(net[1:].state_dict()) != m['expected_head_sha'] and not torch.equal(net[1].proj.weight, w0)
    R.save(R.ROOT / 'cpu_tests.json', dict(status='passed', manifest_sha=R.digest(R.ROOT / 'manifest.json'), formula_head=rep,
                                           actual_head_seed_verified=True, note='frozen DDP numeric tests are head-independent; smoke checks the real gradient against a single-GPU reference'))
    print('PASS', rep, flush=True)


pooling.SpatialPoolEncoder.forward = forward_tokens
R.build_cpu = build_cpu
R.setup = setup
R.save = save
R.source_files = source_files
R.cpu_tests = cpu_tests

if __name__ == '__main__':
    R.main()
