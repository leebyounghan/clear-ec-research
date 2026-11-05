"""Explicit loss-coefficient ablation over the pinned crossfold SSL5 runner.

Only the standardized-log-L1 coefficient changes within [0,0.5]
(default 0; reference 0.5).
The physical-relative coefficient remains 2.5. Data, model, initialization,
optimizer, DDP accumulation, schedule and evaluation use the original runner.
No experiment is selected by this module. Every real command requires an
isolated root and explicit ordinary runner arguments. ``loss_tests`` only uses
synthetic CPU tensors and writes no files or experiment manifests.

An original-runner invocation can read an ablation manifest because the frozen
runner predates this adapter. Such a run lacks the required execution marker:
this adapter's audit rejects it before the original audit can write success.
"""
import argparse
from functools import partial
import json
import math
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import torch

import phase2_crossfold_regression_20261015 as r

b = r.b
LOSS_NAME = 'standardized_log_l1_plus_physical_relative'
RELATIVE_COEFFICIENT = 2.5
ORIGINAL_OBJECTIVE = '0.5 standardized log L1 + 0.5 physical MAPE / 0.2'
ORIGINAL_SOURCE_SHA256 = {
    'phase2_crossfold_regression_20261015.py': '7289b583643c9c74ad300f73e2a0d6debbe38be5d6393b03508437cdc8232ffe',
    'phase2_foundation_train_20261002.py': '20a849cb607c3c5777102e3266c6c1b5d9ba4533f9bfac9f1f6e9d18232dac2f',
    'phase2_foundation_preflight_20261002.py': '196500b67d01ad9423ea4d7d8c1a1e4b2d886df2328d36bfb1e1c94058ca4b1e',
    'phase2_lejepa_regression_20261003.py': 'f66fe903c2360c6e2392b407c55392dc1d7a45a03e624bed0acee8cbc32b33b8',
}
ORIGINAL = {name: getattr(r, name) for name in
            ('prepare', 'verify', 'setup', 'save', 'cpu_tests', 'smoke', 'train', 'audit', 'distributed_backward')}
ORIGINAL_LOSS_FN = b.loss_fn
ARGS = ACTIVE_LOSS = MODEL_FREE_REPORT = None
AUDIT_PRECHECKED = False
AUDIT_BINDINGS = None


def require(value, message):
    if not value:
        raise ValueError(message)


def source_files():
    return [Path(__file__).resolve(), Path(r.__file__).resolve(), Path(b.__file__).resolve(),
            Path(r.prior.__file__).resolve(), Path(r.preflight.__file__).resolve()]


def assert_pinned_sources():
    actual = {path.name: r.digest(path) for path in source_files()[1:]}
    require(actual == ORIGINAL_SOURCE_SHA256, 'Original training implementation SHA256 mismatch')
    require(getattr(ORIGINAL_LOSS_FN, '__name__', None) == 'loss_fn'
            and Path(ORIGINAL_LOSS_FN.__code__.co_filename).resolve() == Path(b.__file__).resolve(),
            'The reference loss was not captured from the original foundation module')
    return actual


def loss_formula(z, targets, mu, sigma, *, standardized_log_l1_coefficient):
    """Original operation/reduction order, with one explicit scalar coefficient."""
    alpha = standardized_log_l1_coefficient
    require(isinstance(alpha, (float, int)) and not isinstance(alpha, bool)
            and math.isfinite(alpha) and 0 <= alpha <= .5, 'Log-L1 coefficient must be finite and within [0,0.5]')
    require(z.ndim == targets.ndim == 2 and z.shape == targets.shape
            and z.shape[0] > 0 and z.shape[1] == 3, 'Expected nonempty matching N-by-3 logits and targets')
    require(mu.shape == sigma.shape == (3,), 'Expected three log means and standard deviations')
    values = (z, targets, mu, sigma)
    require(all(value.is_floating_point() for value in values), 'Loss inputs must have floating dtype')
    require(all(value.device == z.device for value in values), 'Loss inputs must share a device')
    valid = (torch.isfinite(z).all() & torch.isfinite(targets).all() & (targets > 0).all()
             & torch.isfinite(mu).all() & torch.isfinite(sigma).all() & (sigma > 0).all())
    require(bool(valid), 'Loss requires finite inputs, positive targets and positive log standard deviations')
    zt = (targets.log()-mu)/sigma
    value = alpha*(z.float()-zt).abs().mean()+RELATIVE_COEFFICIENT*torch.expm1((z.float()-zt)*sigma).abs().mean()
    require(bool(torch.isfinite(value)), 'Nonfinite loss')
    return value


def recipe_loss(alpha):
    require(isinstance(alpha, (float, int)) and not isinstance(alpha, bool)
            and math.isfinite(alpha) and 0 <= alpha <= .5,
            'Declared log-L1 coefficient must be finite and within [0,0.5]')
    return dict(loss_name=LOSS_NAME, standardized_log_l1_coefficient=float(alpha),
                relative_error_coefficient=RELATIVE_COEFFICIENT,
                objective=f'{alpha:g} standardized log L1 + 2.5 physical relative error')


def assert_active_loss():
    assert_pinned_sources()
    require(ARGS is not None and ACTIVE_LOSS is not None, 'Adapter is not configured')
    recipe_loss(ARGS.log_l1_coefficient)
    require(b.loss_fn is ACTIVE_LOSS and ACTIVE_LOSS.func is loss_formula and ACTIVE_LOSS.args == (),
            'Active loss callable is not the declared adapter')
    require(ACTIVE_LOSS.keywords == {'standardized_log_l1_coefficient': ARGS.log_l1_coefficient},
            'Active loss coefficient differs from the CLI contract')
    require(RELATIVE_COEFFICIENT == 2.5, 'Relative coefficient must remain 2.5')
    require(r.verify is verify and r.setup is setup and r.save is save_hook and r.source_files is source_files,
            'Adapter verification/setup/source hooks are not active')
    for name in ('prepare', 'cpu_tests', 'smoke', 'train', 'audit', 'distributed_backward'):
        require(getattr(r, name) is ORIGINAL[name], 'An inherited execution path was replaced: ' + name)
    require(r.ARGS is ARGS and (r.ROOT, r.SPLIT, r.SSL) == (ARGS.root, ARGS.split, ARGS.ssl_root),
            'Inherited runner arguments differ from adapter arguments')


def execution_contract():
    assert_active_loss()
    return dict(adapter_source_sha256=r.digest(Path(__file__).resolve()),
                base_runner_sha256=ORIGINAL_SOURCE_SHA256[Path(r.__file__).name],
                original_source_sha256=dict(ORIGINAL_SOURCE_SHA256), loss_name=LOSS_NAME,
                standardized_log_l1_coefficient=float(ARGS.log_l1_coefficient),
                relative_error_coefficient=RELATIVE_COEFFICIENT, active_loss_verified=True)


def manifest_loss_contract():
    execution = execution_contract()
    return dict(isolated_root=str(ARGS.root.resolve()), adapter_source_sha256=execution['adapter_source_sha256'],
                original_source_sha256=dict(ORIGINAL_SOURCE_SHA256), original_objective=ORIGINAL_OBJECTIVE,
                required_execution_contract=execution)


def assert_manifest_loss(m):
    recipe_loss(m['recipe'].get('standardized_log_l1_coefficient'))
    require(m.get('loss_ablation') == manifest_loss_contract(), 'Manifest loss/source/root contract differs')
    for key, expected in recipe_loss(ARGS.log_l1_coefficient).items():
        require(m['recipe'].get(key) == expected, 'Manifest loss recipe differs: ' + key)


def check_stage_evidence(m, names):
    expected, manifest_sha = execution_contract(), r.digest(ARGS.root/'manifest.json')
    for name in names:
        record = r.read(ARGS.root/(name+'.json'))
        require(record['status'] == 'passed' and record['manifest_sha'] == manifest_sha,
                'Missing or unbound required stage evidence: ' + name)
        actual = record.get('loss_execution') if name == 'cpu_tests' else record.get('details', {}).get('loss_execution')
        require(actual == expected, 'Stage used an unverified loss implementation: ' + name)
        if name == 'cpu_tests':
            tests = record.get('loss_model_free_tests', {})
            require(tests.get('status') == 'passed', 'Loss CPU evidence is absent')
            require(tests.get('adapter_source_sha256') == expected['adapter_source_sha256']
                    and tests.get('original_source_sha256') == ORIGINAL_SOURCE_SHA256,
                    'Model-free CPU tests used a different implementation')


def verify():
    assert_active_loss()
    declared = r.read(ARGS.root/'manifest.json')
    assert_manifest_loss(declared)
    m = ORIGINAL['verify']()
    assert_manifest_loss(m)
    if ARGS.command in ('smoke', 'train', 'audit'):
        check_stage_evidence(m, ('cpu_tests',) if ARGS.command == 'smoke' else ('cpu_tests', 'smoke'))
    return m


def setup(m):
    assert_manifest_loss(m)
    model, optimizer, mu, sigma, details = ORIGINAL['setup'](m)
    details['loss_execution'] = execution_contract()
    return model, optimizer, mu, sigma, details


def save_new_manifest(path, value):
    """First publication is atomic and refuses to overwrite any existing file."""
    data = (json.dumps(value, indent=2, allow_nan=False)+'\n').encode()
    fd, temporary = tempfile.mkstemp(prefix='.loss-manifest-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def save_hook(path, obj):
    path = Path(path)
    if path.resolve() == (ARGS.root/'manifest.json').resolve():
        require(ARGS.command == 'prepare' and not path.exists(), 'Manifest creation is exclusive to prepare')
        require(obj['recipe']['objective'] == ORIGINAL_OBJECTIVE and 'loss_ablation' not in obj,
                'Original prepare no longer produces the expected reference recipe')
        obj['recipe'].update(recipe_loss(ARGS.log_l1_coefficient))
        obj['loss_ablation'] = manifest_loss_contract()
        obj['limits'].append('Explicit loss-coefficient ablation; only adapter-marked execution is valid evidence')
        assert_manifest_loss(obj)
        save_new_manifest(path, obj)
        return
    if path.resolve() == (ARGS.root/'cpu_tests.json').resolve():
        require(MODEL_FREE_REPORT is not None and MODEL_FREE_REPORT['status'] == 'passed', 'Loss tests were not run')
        obj['loss_execution'] = execution_contract()
        obj['loss_model_free_tests'] = MODEL_FREE_REPORT
    elif path.resolve() == (ARGS.root/'smoke.json').resolve() or path.name == 'result.json':
        require(obj.get('details', {}).get('loss_execution') == execution_contract(),
                'Execution output lacks the actually active loss contract')
    elif path.resolve() == (ARGS.root/'audit.json').resolve():
        require(AUDIT_PRECHECKED and AUDIT_BINDINGS is not None,
                'Loss execution must be checked before publishing an audit')
        obj.update(audit_type='single_model_loss_coefficient_ablation', loss_contract_verified=True,
                   loss_execution=execution_contract())
        obj.update(AUDIT_BINDINGS)
    ORIGINAL['save'](path, obj)


def prepare():
    assert_active_loss()
    require(not (ARGS.root/'manifest.json').exists() and not (ARGS.root/'runs').exists(),
            'Refusing an existing manifest/run root')
    ORIGINAL['prepare']()
    assert_manifest_loss(r.read(ARGS.root/'manifest.json'))


def cpu_tests():
    global MODEL_FREE_REPORT
    MODEL_FREE_REPORT = model_free_tests()
    ORIGINAL['cpu_tests']()


def audit():
    global AUDIT_PRECHECKED, AUDIT_BINDINGS
    m = verify()
    run = ARGS.root/'runs'/f'fold{m["fold"]}'/r.ARCH
    result, complete = r.read(run/'result.json'), r.read(run/'complete.json')
    require(complete['status'] == 'completed' and complete['result_sha'] == r.digest(run/'result.json'),
            'Result is incomplete or unbound')
    require(result['manifest_sha'] == r.digest(ARGS.root/'manifest.json'), 'Result manifest differs')
    require(result.get('details', {}).get('loss_execution') == execution_contract(),
            'Unverified loss execution: original-runner output is not valid ablation evidence')
    AUDIT_BINDINGS = dict(result_sha=r.digest(run/'result.json'), complete_sha=r.digest(run/'complete.json'),
                         checkpoint_sha=result['checkpoint_sha'], predictions_sha=result['predictions_sha'],
                         cpu_tests_sha=r.digest(ARGS.root/'cpu_tests.json'), smoke_sha=r.digest(ARGS.root/'smoke.json'))
    AUDIT_PRECHECKED = True
    try:
        ORIGINAL['audit']()
    finally:
        AUDIT_PRECHECKED = False
        AUDIT_BINDINGS = None


def model_free_tests():
    """Synthetic CPU tensors only: loss/gradient identities and DDP weighting."""
    assert_pinned_sources()
    require(not torch.cuda.is_initialized(), 'Model-free CPU tests cannot run after CUDA initialization')
    tolerances = dict(physical_reference=dict(rtol=2e-5, atol=2e-6),
                      interpolation=dict(rtol=1e-6, atol=1e-7),
                      partial_ddp=dict(rtol=1e-6, atol=1e-7))
    with patch.object(torch.cuda, '_lazy_init', side_effect=RuntimeError('Loss tests forbid CUDA')), \
         patch.object(b, 'build', side_effect=RuntimeError('Loss tests forbid model construction')), \
         patch.object(r, 'build_cpu', side_effect=RuntimeError('Loss tests forbid model construction')):
        torch.set_num_threads(1)
        mu = torch.tensor([8., -1., -.6])
        sigma = torch.tensor([.125, .25, .5])
        targets = torch.tensor([[3000., .35, .55], [2700., .42, .7], [3400., .3, .6]])
        values = torch.tensor([[.4, -.7, 1.1], [-.5, .8, -.9], [.7, .3, -.4]])

        def value_grad(fn, z=values, y=targets, mean=mu, std=sigma):
            leaf = z.detach().clone().requires_grad_(True)
            value = fn(leaf, y, mean, std)
            gradient, = torch.autograd.grad(value, leaf)
            require(torch.isfinite(value).item() and torch.isfinite(gradient).all().item(), 'Nonfinite test loss/gradient')
            return value.detach(), gradient.detach()

        functions = {alpha: partial(loss_formula, standardized_log_l1_coefficient=alpha) for alpha in (0., .25, .5)}
        original_value, original_gradient = value_grad(ORIGINAL_LOSS_FN)
        full_value, full_gradient = value_grad(functions[.5])
        require(torch.equal(full_value, original_value) and torch.equal(full_gradient, original_gradient),
                'Alpha0.5 differs from the actual pinned original loss value or gradient')

        # Independent physical-space reference: no expm1 and no standardized d.
        z64 = values.double().requires_grad_(True)
        y64, mu64, sigma64 = targets.double(), mu.double(), sigma.double()
        physical = torch.exp(mu64+sigma64*z64)
        reference = 2.5*((physical-y64).abs()/y64).mean()
        reference_gradient, = torch.autograd.grad(reference, z64)
        analytic = 2.5*sigma64*(physical.detach()/y64)*torch.sign(physical.detach()-y64)/z64.numel()
        torch.testing.assert_close(reference_gradient, analytic, rtol=1e-12, atol=1e-12)
        pure_value, pure_gradient = value_grad(functions[0.])
        torch.testing.assert_close(pure_value.double(), reference.detach(), **tolerances['physical_reference'])
        torch.testing.assert_close(pure_gradient.double(), reference_gradient, **tolerances['physical_reference'])
        half_value, half_gradient = value_grad(functions[.25])
        torch.testing.assert_close(half_value, (full_value+pure_value)/2, **tolerances['interpolation'])
        torch.testing.assert_close(half_gradient, (full_gradient+pure_gradient)/2, **tolerances['interpolation'])
        for fn in functions.values():
            value, gradient = value_grad(fn, torch.zeros(2, 3), torch.ones(2, 3), torch.zeros(3), sigma)
            require(value.item() == 0 and torch.count_nonzero(gradient).item() == 0, 'Exact-zero loss/subgradient changed')

        invalid_checks = 0
        for alpha in (0., .5):
            cases = []
            for bad in (0., -1., float('nan'), float('inf')):
                bad_targets = targets.clone(); bad_targets[0, 0] = bad
                cases.append((values, bad_targets, mu, sigma))
            for bad in (0., -1., float('nan'), float('inf')):
                bad_sigma = sigma.clone(); bad_sigma[1] = bad
                cases.append((values, targets, mu, bad_sigma))
            bad_mu = mu.clone(); bad_mu[0] = float('nan')
            cases.extend([(values, targets, bad_mu, sigma),
                          (torch.full_like(values, float('inf')), targets, mu, sigma),
                          (torch.full_like(values, 1e6), targets, mu, sigma)])
            for inputs in cases:
                try:
                    functions[alpha](*inputs)
                except ValueError:
                    invalid_checks += 1
                else:
                    raise AssertionError('Invalid/zero target or nonfinite loss was accepted')
        for bad_alpha in (-1., .5001, float('nan'), float('inf')):
            try:
                loss_formula(values, targets, mu, sigma, standardized_log_l1_coefficient=bad_alpha)
            except ValueError:
                invalid_checks += 1
            else:
                raise AssertionError('Invalid coefficient accepted')

        # No nn.Module is built. A shared parameter tensor exercises the same
        # two-rank/two-round sample weights as the pinned distributed_backward.
        sizes, partial_checks = (1, 4, 5, 8, 9, 12, 14, 15, 16), 0
        for alpha, fn in functions.items():
            for n in sizes:
                x = torch.linspace(-.3, .4, n*5, dtype=torch.float64).reshape(n, 5)
                initial = torch.linspace(-.4, .5, 15, dtype=torch.float64).reshape(5, 3)
                target = torch.exp(mu64+sigma64*torch.linspace(-.7, .7, n*3, dtype=torch.float64).reshape(n, 3))
                weight = initial.clone().requires_grad_(True)
                full = fn(x@weight, target, mu64, sigma64)
                expected_gradient, = torch.autograd.grad(full, weight)
                averaged_gradient = torch.zeros_like(weight)
                covered, total_weight = [], 0.
                for rank in (0, 1):
                    local = initial.clone().requires_grad_(True)
                    for j in range(2):
                        start = (2*j+rank)*4
                        count = max(0, min(4, n-start))
                        xx = x[start:start+count] if count else x[:1]
                        yy = target[start:start+count] if count else target[:1]
                        (fn(xx@local, yy, mu64, sigma64)*(2*count/n)).backward()
                        covered.extend(range(start, start+count))
                        total_weight += count/n
                    averaged_gradient += local.grad/2
                require(sorted(covered) == list(range(n)) and abs(total_weight-1) < 1e-12, 'Partial-batch coverage/weight differs')
                torch.testing.assert_close(averaged_gradient, expected_gradient, **tolerances['partial_ddp'])
                partial_checks += 1
        require(not torch.cuda.is_initialized(), 'Loss tests initialized CUDA')
    return dict(status='passed', original_alpha05_value_gradient_exact=True,
                alpha0_independent_physical_value_gradient=True, physical_analytic_gradient=True,
                alpha025_interpolation=True, exact_zero_subgradient=True,
                invalid_input_rejections=invalid_checks, partial_batch_sizes=list(sizes),
                partial_batch_gradient_cases=partial_checks, includes_final14_and_dummy9=True,
                tolerances=tolerances, tolerance_fixed_before_observation=True,
                model_constructed=False, cuda_initialized=False,
                original_source_sha256=dict(ORIGINAL_SOURCE_SHA256),
                adapter_source_sha256=r.digest(Path(__file__).resolve()))


def configure(args):
    global ARGS, ACTIVE_LOSS
    assert_pinned_sources()
    require(b.loss_fn is ORIGINAL_LOSS_FN, 'Loss was already replaced before adapter configuration')
    ARGS = args
    ACTIVE_LOSS = partial(loss_formula, standardized_log_l1_coefficient=args.log_l1_coefficient)
    r.ARGS, r.ROOT, r.SPLIT, r.SSL = args, args.root, args.split, args.ssl_root
    r.source_files, r.verify, r.setup, r.save = source_files, verify, setup, save_hook
    r.EXPECTED_REUSED_SOURCES = dict(ORIGINAL_SOURCE_SHA256)
    b.loss_fn = ACTIVE_LOSS
    assert_active_loss()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'cpu_tests', 'cpu-tests', 'smoke', 'train', 'audit', 'loss_tests'])
    for name in ('root', 'split', 'ssl-root'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--fold', type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--head-lr', type=float)
    parser.add_argument('--clip', type=float)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--schedule-epochs', type=int, default=20)
    parser.add_argument('--deadline-utc')
    parser.add_argument('--log-l1-coefficient', type=float, default=0.)
    args = parser.parse_args()
    if args.command == 'loss_tests':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        print(json.dumps(model_free_tests(), allow_nan=False), flush=True)
        return
    args.command = args.command.replace('-', '_')
    for name in ('root', 'split', 'ssl_root', 'fold', 'head_lr', 'clip', 'deadline_utc'):
        if getattr(args, name) is None:
            parser.error('--'+name.replace('_', '-')+' is required for an experiment command')
    if not math.isfinite(args.log_l1_coefficient) or not 0 <= args.log_l1_coefficient <= .5:
        parser.error('log-l1-coefficient must be finite and within [0,0.5]')
    if args.seed < 0 or args.seed >= 2**32-1 or args.fold < 0:
        parser.error('seed must be in [0,2**32-2], fold nonnegative')
    if (args.epochs, args.schedule_epochs) != (20, 20):
        parser.error('This loss-only option keeps epochs20 and schedule-epochs20')
    if not all(math.isfinite(value) and value > 0 for value in (args.head_lr, args.clip)):
        parser.error('head-lr and clip must be finite and positive')
    r.parse_deadline(args.deadline_utc)
    configure(args)
    commands = dict(prepare=prepare, cpu_tests=cpu_tests, smoke=ORIGINAL['smoke'],
                    train=ORIGINAL['train'], audit=audit)
    commands[args.command]()


if __name__ == '__main__':
    main()
