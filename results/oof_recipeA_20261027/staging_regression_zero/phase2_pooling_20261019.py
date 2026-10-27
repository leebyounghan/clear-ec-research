"""Manifest-selected spatial pooling; no labels, calibration, or extra views."""
import torch
from torch import nn

MODES = {
    'mean': ('mean',), 'std': ('std',), 'min': ('min',), 'max': ('max',),
    'mean_std': ('mean', 'std'),
    'mean_std_min_max': ('mean', 'std', 'min', 'max'),
}
STD_EPS = 1e-12


def pool_tokens(tokens, mode):
    if mode not in MODES:
        raise ValueError(mode)
    if tokens.ndim != 3 or tokens.shape[1] < 1:
        raise ValueError('Expected nonempty B,N,C patch tokens')
    # Preserve historical mean exactly. Accumulate variance in float32.
    values = []
    for stat in MODES[mode]:
        if stat == 'mean':
            value = tokens.mean(1)
        elif stat == 'std':
            value = (tokens.float().var(1, unbiased=False) + STD_EPS).sqrt().to(tokens.dtype)
        elif stat == 'min':
            value = tokens.amin(1)
        else:
            value = tokens.amax(1)
        values.append(value)
    return values[0] if len(values) == 1 else torch.cat(values, dim=-1)


class SpatialPoolEncoder(nn.Module):
    def __init__(self, original, mode):
        super().__init__()
        if mode not in MODES:
            raise ValueError(mode)
        self.original = original
        self.mode = mode

    def forward(self, x):
        return pool_tokens(self.original.forward_features(x)['x_norm_patchtokens'], self.mode)


def cpu_contract_tests():
    x = torch.tensor([[[1., 3.], [3., 7.]]], requires_grad=True)
    expected = {'mean': [[2.,5.]], 'std': [[1.,2.]], 'min': [[1.,3.]], 'max': [[3.,7.]]}
    for mode, stats in MODES.items():
        out = pool_tokens(x, mode)
        ref = torch.cat([torch.tensor(expected[s]) for s in stats], -1)
        torch.testing.assert_close(out, ref)
        torch.testing.assert_close(out, pool_tokens(x.flip(1), mode))
        grad, = torch.autograd.grad(out.sum(), x, retain_graph=True)
        assert torch.isfinite(grad).all()
    constant = torch.ones(2, 4, 3, requires_grad=True)
    pool_tokens(constant, 'mean_std_min_max').sum().backward()
    assert torch.isfinite(constant.grad).all()
    assert torch.equal(pool_tokens(x, 'mean'), x.mean(1))
    return {'modes': list(MODES), 'population_variance': True, 'std_epsilon': STD_EPS,
            'known_values_permutation_gradients_constant_tokens': 'passed'}
