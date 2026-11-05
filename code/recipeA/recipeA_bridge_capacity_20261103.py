"""Recipe A bridge-capacity test (ceiling meta-analysis, 2026-11-03).

Everything is the frozen phase2_pooling_regression_20261019 runner with std pooling (the recipe A member runner)
except one change, chosen by BRIDGE_VARIANT and pinned in <root>/bridge_variant.json at the first call:
  enc_lr5x  : encoder learning rate 5e-5 instead of 1e-5 (written into the manifest recipe; the schedule, head,
              data and seed are unchanged).
  token_mlp : a per-token residual MLP between the encoder and the std pooling, t -> t + W2 GELU(W1 t), 768 wide,
              W2 and its bias zero-initialised so the network starts exactly as recipe A; its parameters train at
              the head learning rate (they are new), the encoder keeps 1e-5. Std of a learned per-token function is
              a richer statistic than std of the raw tokens (cell-shape statistics need a non-linear token map).
Why: the train-row error of recipe A is almost as high as its validation error on CV and HEX (fold 3: train
9.59 / 9.75 vs validation 10.14 / 10.27; fold 2: 9.66 / 9.76 vs 10.82 / 10.31), i.e. the model underfits the
training labels. Either the encoder barely moves at 1e-5 (enc_lr5x) or the image-to-statistic bridge cannot
express the statistic (token_mlp). Read: the runner's online train-probe metrics (256 train rows, eval mode) and
the validation predictions per epoch, against the matched seed-46 recipe A root of the same fold.
"""
import json, os
from pathlib import Path

import torch

import phase2_pooling_20261019 as pooling
import phase2_pooling_regression_20261019 as R

WRAPPER = Path(__file__).resolve()
VARIANTS = {'enc_lr5x': dict(encoder_lr=5e-5, token_mlp=False), 'token_mlp': dict(encoder_lr=1e-5, token_mlp=True)}
FROZEN_SAVE = R.save
FROZEN_SETUP = R.setup
FROZEN_SOURCES = R.source_files
FROZEN_CPU_TESTS = R.cpu_tests
FROZEN_INIT = pooling.SpatialPoolEncoder.__init__


def variant():
    env = os.environ.get('BRIDGE_VARIANT')
    pin = R.ROOT / 'bridge_variant.json'
    if pin.exists():
        name = json.loads(pin.read_text())['name']
        assert env is None or env == name, f'BRIDGE_VARIANT {env} differs from pinned {name}'
    else:
        assert env in VARIANTS, 'set BRIDGE_VARIANT for the first call'
        name = env
        R.ROOT.mkdir(parents=True, exist_ok=True)
        pin.write_text(json.dumps(dict(name=name, **VARIANTS[name])))
    return dict(name=name, **VARIANTS[name])


def init(self, original, mode):
    FROZEN_INIT(self, original, mode)
    if variant()['token_mlp']:
        self.phi = torch.nn.Sequential(torch.nn.Linear(768, 768), torch.nn.GELU(), torch.nn.Linear(768, 768))
        torch.nn.init.zeros_(self.phi[2].weight)
        torch.nn.init.zeros_(self.phi[2].bias)


def forward(self, x):
    t = self.original.forward_features(x)['x_norm_patchtokens']
    if getattr(self, 'phi', None) is not None:
        t = t + self.phi(t)
    return pooling.pool_tokens(t, self.mode)


def save(path, obj):
    if Path(path).name == 'manifest.json' and isinstance(obj, dict) and 'recipe' in obj:
        v = variant()
        obj['recipe']['encoder_lr'] = v['encoder_lr']
        obj['recipe']['bridge_variant'] = v
        if v['token_mlp']:
            obj['recipe']['head'] = 'token residual MLP 768-GELU-768 (zero-init out) -> ' + obj['recipe']['head']
    return FROZEN_SAVE(path, obj)


def setup(m):
    net, opt, mu, sigma, details = FROZEN_SETUP(m)
    assert m['recipe']['bridge_variant']['name'] == variant()['name']
    if variant()['token_mlp']:
        phi = list(net[0].phi.parameters())
        ids = {id(p) for p in phi}
        enc = [p for p in net[0].parameters() if id(p) not in ids]
        r = m['recipe']
        opt = torch.optim.AdamW([{'params': enc, 'lr': r['encoder_lr']}, {'params': list(net[1:].parameters()) + phi, 'lr': r['head_lr']}],
                                weight_decay=r['weight_decay'], fused=True)
        details['token_mlp_trainable'] = sum(p.numel() for p in phi)
    details['bridge_variant'] = variant()
    return net, opt, mu, sigma, details


def source_files():
    return FROZEN_SOURCES() + [WRAPPER]


def bridge_tests():
    v = variant()
    enc = pooling.SpatialPoolEncoder.__new__(pooling.SpatialPoolEncoder)
    torch.nn.Module.__init__(enc)
    out = {}
    if v['token_mlp']:
        phi = torch.nn.Sequential(torch.nn.Linear(768, 768), torch.nn.GELU(), torch.nn.Linear(768, 768))
        torch.nn.init.zeros_(phi[2].weight); torch.nn.init.zeros_(phi[2].bias)
        t = torch.randn(2, 10, 768)
        torch.testing.assert_close(pooling.pool_tokens(t + phi(t), 'std'), pooling.pool_tokens(t, 'std'))   # identity at init
        (pooling.pool_tokens(t + phi(t), 'std').sum()).backward()
        assert phi[2].weight.grad.abs().sum() > 0                                                        # output layer learns
        out['identity_at_init'] = True
    out.update(v)
    return out


def cpu_tests():
    print({'bridge_tests': bridge_tests()}, flush=True)
    return FROZEN_CPU_TESTS()


pooling.SpatialPoolEncoder.__init__ = init
pooling.SpatialPoolEncoder.forward = forward
R.save = save
R.setup = setup
R.source_files = source_files
R.cpu_tests = cpu_tests

if __name__ == '__main__':
    R.main()
