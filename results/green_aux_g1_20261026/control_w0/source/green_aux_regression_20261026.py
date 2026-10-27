"""GREEN-G1 matched A/B: fold-clean regression with a train-only green point auxiliary.

The frozen runner phase2_backbone_ssl_regression_20261020 supplies the manifest,
data, regression loss, DDP micro-batch split, schedule, clipping, validation,
checkpoint format and audit unchanged. This wrapper adds a linear 768->1
patch-token point head (initialised exactly like green_point_probe_20261023)
trained with ROI-masked Gaussian BCE on the 20 fold4-train green images. Each
rank appends one green image to its second micro-batch, so the encoder still runs
once per micro-batch. aux_weight 0 is the matched control through the identical
code path. The 5 validation green images are opened only after terminal epoch 20.
"""
import argparse
from contextlib import nullcontext
import datetime
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

import phase2_backbone_ssl_regression_20261020 as R
import phase2_pooling_20261019 as pooling
import phase2_foundation_train_20261002 as b
import phase2_lejepa_regression_20261003 as prior
import green_point_spatial_20261023 as spatial
import green_point_probe_head_20261023 as probe
from phase2_foundation_preflight_20261002 import digest, save

ANN = Path('datasets/clear_ec_official/green_eval/annotations.json')
BINDING = Path('experiments/green_point_probe_20261023/data_binding.json')
PROBE_RESULT = Path('experiments/green_point_probe_20261023/result.json')
WEIGHTS = (0.0, 0.1, 0.3)
WIDTH, HEIGHT = 1296, 972
GRID = (124, 164)
FLIPS = ((False, False), (True, False), (False, True), (True, True))
ARGS = ROOT = None


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def own_sources():
    return [Path(__file__).resolve(), Path(spatial.__file__).resolve(), Path(probe.__file__).resolve()]


def build_point_head(seed):
    """Same weights as probe.build_head(seed), touching only (and restoring) the CPU generator."""
    state = torch.random.get_rng_state()
    try:
        torch.random.default_generator.manual_seed(seed)
        head = torch.nn.Linear(768, 1)
    finally:
        torch.random.set_rng_state(state)
    return head


def flip_annotation(points, roi, horizontal, vertical):
    """Index-coordinate flips matching torch.flip of the native 972x1296 frame."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2).copy()
    x0, y0, x1, y1 = (float(v) for v in roi)
    if horizontal:
        p[:, 0] = WIDTH-1-p[:, 0]
        x0, x1 = WIDTH-1-x1, WIDTH-1-x0
    if vertical:
        p[:, 1] = HEIGHT-1-p[:, 1]
        y0, y1 = HEIGHT-1-y1, HEIGHT-1-y0
    return p, [x0, y0, x1, y1]


def flip_image(x, flip):
    horizontal, vertical = FLIPS[flip]
    if horizontal:
        x = torch.flip(x, (-1,))
    if vertical:
        x = torch.flip(x, (-2,))
    return x


def green_split(ids, tr, va):
    binding = read(BINDING)
    names = dict(train=list(binding['train_ids']), validation=list(binding['validation_ids']))
    assert len(names['train']) == 20 and len(names['validation']) == 5
    assert not set(names['train']) & set(names['validation'])
    lookup = {name: i for i, name in enumerate(ids)}
    train = np.array([lookup[n] for n in names['train']], dtype=np.int64)
    validation = np.array([lookup[n] for n in names['validation']], dtype=np.int64)
    assert np.isin(train, tr).all(), 'Green training images must be fold training images'
    assert np.isin(validation, va).all(), 'Green validation images must be fold validation images'
    return names, train, validation


def green_targets(names, ann):
    """[image, flip] -> (target, support) on the 124x164 patch grid."""
    targets = np.zeros((len(names), len(FLIPS), *GRID), dtype=np.float32)
    support = np.zeros((len(names), len(FLIPS), *GRID), dtype=bool)
    for i, name in enumerate(names):
        for f, (horizontal, vertical) in enumerate(FLIPS):
            points, roi = flip_annotation(ann[name]['markers_mha'], ann[name]['roi_box_mha'], horizontal, vertical)
            targets[i, f], support[i, f] = spatial.point_target(points, roi, GRID)
    return targets, support


def green_images(indices, y, total):
    ds = b.Frames(indices, y, total)
    return torch.stack([ds[i][0] for i in range(len(indices))])


def green_permutation(seed, epoch, n=20):
    return np.random.default_rng(np.random.SeedSequence([seed, 7, epoch])).permutation(n)


def green_choice(permutation, seed, epoch, step, rank):
    """Rank-specific green image and flip for one update; independent of every global RNG."""
    image = int(permutation[(2*step+rank) % len(permutation)])
    flip = int(np.random.default_rng(np.random.SeedSequence([seed, 11, epoch, step, rank])).integers(len(FLIPS)))
    return image, flip


class AuxNet(torch.nn.Module):
    def __init__(self, net, head):
        super().__init__()
        self.net, self.point = net, head

    def forward(self, x, n_green=0):
        encoder = self.net[0]
        tokens = encoder.original.forward_features(x)['x_norm_patchtokens']
        n = len(x)-n_green
        z = self.net[1:](pooling.pool_tokens(tokens[:n], encoder.mode)) if n else None
        if not n_green:
            return z, None
        with torch.autocast('cuda', enabled=False):
            logits = self.point(tokens[n:].to(self.point.weight.dtype)).squeeze(-1)
        return z, logits


def aux_loss(logits, target, support):
    assert support.dtype == torch.bool and support.any()
    logits = logits.reshape(target.shape)
    return F.binary_cross_entropy_with_logits(logits[support], target[support].to(logits.dtype))


def distributed_backward_aux(ddp, x, t, mu, sigma, rank, green, weight):
    """Frozen runner split; one green image joins this rank's second (synchronising) micro-batch."""
    gx, gt, gs = green
    n = len(t)
    stats = torch.zeros(8, device=mu.device, dtype=torch.float64)
    for j in range(2):
        start = (2*j+rank)*4
        count = max(0, min(4, n-start))
        xx, yy = (x[start:start+count], t[start:start+count]) if count else (x[:1], t[:1])
        g = int(j == 1)
        if g:
            xx = torch.cat([xx, gx])
        with ddp.no_sync() if j == 0 else nullcontext():
            with torch.autocast('cuda', dtype=torch.bfloat16):
                z, logits = ddp(b.inputs(xx, R.ARCH), g)
                loss = b.loss_fn(z, yy.cuda(non_blocking=True), mu, sigma)
                aux = aux_loss(logits, gt, gs) if g else None
            if not torch.isfinite(loss) or (g and not torch.isfinite(aux)):
                raise ValueError('Nonfinite loss')
            if count:
                with torch.no_grad():
                    zt = (yy.cuda(non_blocking=True).log()-mu)/sigma
                    rel = torch.expm1((z.float()-zt)*sigma).abs()
                    stats[0] += loss.detach().double()*count
                    stats[1:4] += rel.double().sum(0)*100
                    stats[4:7] += (z.float()-zt).abs().double().sum(0)
            total = loss*(2*count/n)
            if g:
                stats[7] += aux.detach().double()
                total = total+weight*aux
            total.backward()
    return stats


def verify():
    m = R.verify()
    a = read(ROOT/'aux_manifest.json')
    assert a['base_manifest_sha'] == digest(ROOT/'manifest.json')
    assert a['aux_weight'] == ARGS.aux_weight
    for path, expected in a['hashes'].items():
        assert digest(path) == expected, f'Frozen aux input/source changed: {path}'
    for source in own_sources():
        assert digest(source) == a['hashes'][str(ROOT/'source'/source.name)]
    return m, a


def setup_aux(m, a):
    net, opt, mu, sigma, details = R.setup(m)
    head = build_point_head(m['seed'])
    assert prior.state_sha(head.state_dict()) == a['point_head_initial_sha']
    head = head.cuda()
    opt.add_param_group({'params': list(head.parameters()), 'lr': m['recipe']['head_lr'],
                         'weight_decay': m['recipe']['weight_decay']})
    details.update(aux_weight=a['aux_weight'], point_head_trainable=sum(p.numel() for p in head.parameters()),
                   point_head_initial_sha=a['point_head_initial_sha'])
    return AuxNet(net, head), opt, mu, sigma, details


def gradient_shares(module, x, t, green, mu, sigma):
    """Separate encoder gradient norms of the regression and point losses on one fixed batch."""
    gx, gt, gs = green
    params = list(module.net[0].parameters())
    with torch.autocast('cuda', dtype=torch.bfloat16):
        z, logits = module(b.inputs(torch.cat([x, gx]), R.ARCH), 1)
        reg = b.loss_fn(z, t.cuda(), mu, sigma)
        aux = aux_loss(logits, gt, gs)
    gr = torch.autograd.grad(reg, params, retain_graph=True, allow_unused=True)
    ga = torch.autograd.grad(aux, params, allow_unused=True)
    pairs = [(r.float(), g.float()) for r, g in zip(gr, ga) if r is not None and g is not None]
    nr = math.sqrt(sum(r.square().sum().item() for r in gr if r is not None))
    na = math.sqrt(sum(g.square().sum().item() for g in ga if g is not None))
    dot = sum((r*g).sum().item() for r, g in pairs)
    return dict(regression_loss=reg.item(), point_loss=aux.item(), regression_encoder_grad_norm=nr,
                point_encoder_grad_norm=na, point_to_regression_norm_ratio=na/max(nr, 1e-30),
                cosine=dot/max(nr*na, 1e-30), mode='eval-mode encoder, one fixed batch, unweighted point loss')


def green_point_eval(module, indices, names, ann, y, total, prior_value):
    """Frozen-probe evaluation contract: ROI support, threshold .5, NMS16, one-to-one match radius12."""
    module.eval()
    xs = green_images(indices, y, total)
    rows, probabilities = [], []
    with torch.inference_mode():
        for i, name in enumerate(names):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                tokens = module.net[0].original.forward_features(b.inputs(xs[i:i+1], R.ARCH))['x_norm_patchtokens']
            p = module.point(tokens[0].float()).squeeze(-1).sigmoid().reshape(GRID).cpu().numpy()
            _, support = spatial.point_target(ann[name]['markers_mha'], ann[name]['roi_box_mha'], GRID)
            points = ann[name]['markers_mha']
            found = probe.peaks(p, support)
            constant = probe.peaks(np.full_like(p, prior_value), support)
            rows.append(dict(id=name, point=spatial.match_points(found, points),
                             baseline=spatial.match_points(constant, points),
                             annotated=len(points), predicted=len(found),
                             count_absolute_error=abs(len(found)-len(points))))
            probabilities.append(p)
    return rows, np.stack(probabilities)


def prepare():
    assert ARGS.aux_weight in WEIGHTS, 'Only pre-registered aux weights'
    assert (ARGS.fold, ARGS.seed, ARGS.pooling, ARGS.ssl_epochs, ARGS.deadline_utc) == (4, 46, 'mean', 5, 'none')
    if (ROOT/'aux_manifest.json').exists():
        raise FileExistsError('Frozen aux manifest already exists')
    R.prepare()
    frozen = []
    for source in own_sources():
        destination = ROOT/'source'/source.name
        if destination.exists():
            assert digest(destination) == digest(source), 'Frozen source differs'
        else:
            with source.open('rb') as src, destination.open('xb') as dst:
                shutil.copyfileobj(src, dst)
        frozen.append(destination)
    ids, y, tr, va = R.arrays()
    names, gtr, gva = green_split(ids, tr, va)
    targets, support = green_targets(names['train'], read(ANN))
    head = build_point_head(ARGS.seed)
    files = [ANN, BINDING, PROBE_RESULT, *own_sources(), *frozen]
    a = dict(created_utc=now(), base_manifest_sha=digest(ROOT/'manifest.json'), aux_weight=ARGS.aux_weight,
             pre_registered_weights=list(WEIGHTS), green_train_ids=names['train'],
             green_validation_ids=names['validation'], green_train_indices=gtr.tolist(),
             green_validation_indices=gva.tolist(), target_sha=R.array_sha(targets),
             support_sha=R.array_sha(support), point_head_initial_sha=prior.state_sha(head.state_dict()),
             point_head='Linear 768->1 on final normed patch tokens in float32; init identical to green_point_probe_20261023',
             point_loss='BCE-with-logits against max-Gaussian sigma8 target, annotated ROI support cells only (frozen probe formula)',
             green_sampling=('one train-green image per rank per update, appended to the second micro-batch; '
                             'epoch permutation SeedSequence([seed,7,epoch]); flip SeedSequence([seed,11,epoch,step,rank]) '
                             'uniform over 4 H/V flips with exactly transformed points and ROI'),
             objective='frozen regression loss + aux_weight * rank mean of point loss',
             optimizer='point head in a third AdamW group with the head LR schedule and recipe weight decay',
             clipping='global norm over encoder, regression head and point head',
             validation_green='5 fold4-validation green images, only after terminal epoch20; frozen probe peaks/NMS/matching; constant prior from train20',
             primary='terminal epoch20 full-validation regression MAPE versus aux_weight0 control',
             limits=['Single fold4 seed46 screen; reused development fold',
                     '20 training and 5 validation green images; high variance point metrics',
                     'Green ROI/points are not cell masks; original TIFF completeness unverified',
                     'Patient independence and near-duplicate leakage unresolved'],
             hashes={str(p): digest(p) for p in files})
    save(ROOT/'aux_manifest.json', a)
    print(json.dumps(dict(status='prepared', root=str(ROOT), aux_weight=ARGS.aux_weight)), flush=True)


class FakeOriginal(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Conv2d(3, 768, 8, 8)

    def forward_features(self, x):
        return {'x_norm_patchtokens': self.proj(x).flatten(2).transpose(1, 2)}


def aux_equivalence_tests(m):
    net, _ = R.build_cpu(m['seed'])
    fake = FakeOriginal()
    net[0] = pooling.SpatialPoolEncoder(fake, ARGS.pooling)
    net = net.double()
    head = build_point_head(m['seed']).double()
    aux = AuxNet(net, head)
    gen = torch.Generator().manual_seed(5)
    x = torch.randn(3, 3, 16, 24, generator=gen, dtype=torch.float64)
    g = torch.randn(1, 3, 16, 24, generator=gen, dtype=torch.float64)
    target = torch.rand(1, 2, 3, generator=gen, dtype=torch.float64)
    support = torch.zeros(1, 2, 3, dtype=torch.bool)
    support[0, :, 1:] = True
    z, logits = aux(torch.cat([x, g]), 1)
    torch.testing.assert_close(z, net(x), rtol=1e-10, atol=1e-12)
    assert logits.shape == (1, 6)

    def grads(weight):
        aux.zero_grad(set_to_none=True)
        out, lg = aux(torch.cat([x, g]), 1)
        (out.square().mean()+weight*aux_loss(lg, target, support)).backward()
        return [p.grad.clone() for p in fake.parameters()], [p.grad.clone() for p in head.parameters()]
    aux.zero_grad(set_to_none=True)
    net(x).square().mean().backward()
    reference = [p.grad.clone() for p in fake.parameters()]
    encoder0, head0 = grads(0.)
    for a, c in zip(encoder0, reference):
        torch.testing.assert_close(a, c, rtol=1e-10, atol=1e-12)
    assert all((p == 0).all() for p in head0)
    encoder1, head1 = grads(.3)
    assert any(not torch.allclose(a, c) for a, c in zip(encoder1, reference))
    assert all(torch.isfinite(p).all() and p.abs().max() > 0 for p in head1)
    lg = logits.detach().reshape(1, 2, 3).clone()
    t2 = target.clone()
    lg[~support], t2[~support] = 99., 1.
    assert torch.equal(aux_loss(logits.detach(), target, support), aux_loss(lg.reshape(1, -1), t2, support))
    return ['regression_output_unchanged_by_appended_green', 'weight0_encoder_gradients_equal_regression_only',
            'weight0_point_head_gradient_zero', 'weight_positive_changes_encoder_gradients',
            'point_loss_ignores_cells_outside_ROI']


def real_encoder_test(m):
    net, _ = R.build_cpu(m['seed'])
    checkpoint = torch.load(R.SSL/'run/last.pt', map_location='cpu', weights_only=True)
    net[0].original.load_state_dict(checkpoint['encoder'], strict=True)
    del checkpoint
    aux = AuxNet(net, build_point_head(m['seed'])).eval()
    x = torch.rand(3, 3, 96, 128, generator=torch.Generator().manual_seed(9))
    with torch.inference_mode():
        z, logits = aux(x, 1)
        alone, _ = aux(x[:2])
    assert z.shape == (2, 3) and logits.shape == (1, 12*16)
    assert torch.isfinite(z).all() and torch.isfinite(logits).all()
    torch.testing.assert_close(z, alone, rtol=1e-4, atol=1e-5)
    assert GRID == (992//8, 1312//8)
    return dict(input=[3, 3, 96, 128], token_grid=[12, 16], full_frame_grid=list(GRID),
                regression_shape=list(z.shape), point_shape=list(logits.shape))


def cpu_tests():
    m, a = verify()
    if (ROOT/'aux_cpu_tests.json').exists():
        raise FileExistsError('Aux CPU test evidence already exists')
    if not (ROOT/'cpu_tests.json').exists():
        R.cpu_tests()
    assert read(ROOT/'cpu_tests.json')['status'] == 'passed'
    checks = dict(spatial=spatial.cpu_tests(), probe_head=probe.cpu_tests())
    state = torch.random.get_rng_state()
    head = build_point_head(m['seed'])
    assert torch.equal(state, torch.random.get_rng_state()), 'Point head init must not move the global RNG'
    reference = probe.build_head(m['seed'])
    assert all(torch.equal(p, q) for p, q in zip(head.parameters(), reference.parameters()))
    assert prior.state_sha(head.state_dict()) == a['point_head_initial_sha']
    ids, y, tr, va = R.arrays()
    names, gtr, gva = green_split(ids, tr, va)
    assert gtr.tolist() == a['green_train_indices'] and gva.tolist() == a['green_validation_indices']
    assert not np.isin(gva, tr).any() and not np.isin(gtr, va).any()
    points = np.array([[0, 0], [1295, 971], [634, 329], [10, 900]], dtype=np.float64)
    frame = torch.zeros(1, HEIGHT, WIDTH)
    for px, py in points.astype(int):
        frame[0, py, px] = 1
    for f, (horizontal, vertical) in enumerate(FLIPS):
        moved, _ = flip_annotation(points, [0, 0, WIDTH-1, HEIGHT-1], horizontal, vertical)
        found = sorted(map(tuple, torch.nonzero(flip_image(frame, f)[0]).flip(1).tolist()))
        assert found == sorted(map(tuple, moved.astype(int).tolist())), f
    ann = read(ANN)
    targets, support = green_targets(names['train'], ann)
    assert R.array_sha(targets) == a['target_sha'] and R.array_sha(support) == a['support_sha']
    assert targets.shape == (20, 4, *GRID) and support.any(axis=(2, 3)).all() and np.isfinite(targets).all()
    for i, name in enumerate(names['train']):
        for f in range(1, 4):
            assert abs(targets[i, f][support[i, f]].sum()-targets[i, 0][support[i, 0]].sum()) < .05*targets[i, 0][support[i, 0]].sum()+1
    numpy_state = np.random.get_state()
    updates = math.ceil(len(tr)/16)
    for epoch in (0, 1, 19):
        permutation = green_permutation(m['seed'], epoch)
        assert sorted(permutation.tolist()) == list(range(20))
        used = [green_choice(permutation, m['seed'], epoch, s, r) for s in range(updates) for r in (0, 1)]
        counts = np.bincount([u[0] for u in used], minlength=20)
        assert counts.min() >= (2*updates)//20 and counts.max() <= -(-2*updates//20)
        assert used == [green_choice(permutation, m['seed'], epoch, s, r) for s in range(updates) for r in (0, 1)]
        assert len({u[1] for u in used}) == 4
        assert green_choice(permutation, m['seed'], epoch, 0, 0) != green_choice(permutation, m['seed'], epoch, 0, 1)
    assert not np.array_equal(green_permutation(m['seed'], 0), green_permutation(m['seed'], 1))
    assert all(np.array_equal(u, v) if isinstance(u, np.ndarray) else u == v
               for u, v in zip(numpy_state, np.random.get_state())), 'Green schedule must not move the global RNG'
    checks['aux_equivalence'] = aux_equivalence_tests(m)
    checks['real_encoder_small_input'] = real_encoder_test(m)
    save(ROOT/'aux_cpu_tests.json', dict(status='passed', manifest_sha=digest(ROOT/'manifest.json'),
         aux_manifest_sha=digest(ROOT/'aux_manifest.json'), checks=checks,
         verified=['green_ids_inside_fold_partitions', 'flip_coordinate_identity', 'target_hashes',
                   'schedule_coverage_determinism_rank_difference', 'global_rng_untouched',
                   'point_head_init_matches_frozen_probe']))
    print('PASS: aux split, flips, targets, schedule, equivalence, real encoder small input', flush=True)


def smoke():
    m, a = verify()
    rank = R.init_dist()
    if (ROOT/'smoke.json').exists():
        raise FileExistsError('Smoke already exists')
    for name in ('cpu_tests.json', 'aux_cpu_tests.json'):
        assert read(ROOT/name)['status'] == 'passed' and read(ROOT/name)['manifest_sha'] == digest(ROOT/'manifest.json')
    aux, opt, mu, sigma, details = setup_aux(m, a)
    ddp = DDP(aux, device_ids=[rank], broadcast_buffers=False, find_unused_parameters=True)
    ids, y, tr, va = R.arrays()
    names, gtr, _ = green_split(ids, tr, va)
    ann = read(ANN)
    ds = R.Frames(tr[:9], y, len(ids))
    data = [ds[i] for i in range(9)]
    x = torch.stack([v[0] for v in data])
    t = torch.stack([v[1] for v in data])
    images = green_images(gtr[:2], y, len(ids))
    targets, support = green_targets(names['train'][:2], ann)
    greens = [(flip_image(images[r:r+1], r+1), torch.from_numpy(targets[r, r+1][None]).cuda(),
               torch.from_numpy(support[r, r+1][None]).cuda()) for r in (0, 1)]
    weight = a['aux_weight']
    ddp.eval()
    opt.zero_grad(set_to_none=True)
    distributed_backward_aux(ddp, x, t, mu, sigma, rank, greens[rank], weight)
    evidence = [None]
    if rank == 0:
        ref, _, rmu, rsigma, _ = setup_aux(m, a)
        ref.eval()
        ref.zero_grad(set_to_none=True)
        for start in range(0, 9, 4):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                z, _ = ref(b.inputs(x[start:start+4], R.ARCH))
                loss = b.loss_fn(z, t[start:start+4].cuda(), rmu, rsigma)
            (loss*len(t[start:start+4])/9).backward()
        for gx, gt, gs in greens:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                _, logits = ref(b.inputs(gx, R.ARCH), 1)
                point = aux_loss(logits, gt, gs)
            (weight*point/2).backward()
        error = norm = maximum = 0.
        for (name, p), (other, q) in zip(aux.named_parameters(), ref.named_parameters()):
            assert name == other and (p.grad is None) == (q.grad is None)
            if p.grad is not None:
                diff = (p.grad-q.grad).float()
                error += diff.square().sum().item()
                norm += q.grad.float().square().sum().item()
                maximum = max(maximum, diff.abs().max().item())
        relative = math.sqrt(error/max(norm, 1e-30))
        shares = gradient_shares(ref, x[:4], t[:4], greens[0], rmu, rsigma)
        evidence[0] = dict(relative_gradient_l2=relative, max_abs_gradient_error=maximum, passed=relative < .002,
                           includes_final9_samples_zero_weight_dummy_and_two_rank_green=True,
                           initial_gradient_shares=shares)
        del ref
    dist.broadcast_object_list(evidence, src=0)
    assert evidence[0]['passed'], evidence[0]
    net = aux.net
    before = R.block_hashes(net)
    assert len(before) == 12
    head_before = prior.state_sha(net[1:].state_dict())
    point_before = prior.state_sha(aux.point.state_dict())
    reached = {R.block_key(n) for n, p in net[0].original.named_parameters()
               if R.block_key(n) and p.grad is not None and p.grad.abs().max().item() > 0}
    assert set(before) == reached
    assert any('patch_embed' in n and p.grad is not None and p.grad.abs().max().item() > 0
               for n, p in net[0].original.named_parameters())
    point_grad = max(p.grad.abs().max().item() for p in aux.point.parameters())
    assert (point_grad > 0) if weight > 0 else (point_grad == 0), point_grad
    b.finite_grad(aux)
    torch.nn.utils.clip_grad_norm_(aux.parameters(), m['recipe']['gradient_clip'], error_if_nonfinite=True)
    opt.step()
    after = R.block_hashes(net)
    assert all(before[k] != after[k] for k in before), 'All encoder blocks must actually update'
    assert prior.state_sha(net[1:].state_dict()) != head_before
    if weight > 0:
        assert prior.state_sha(aux.point.state_dict()) != point_before
    hashes = [None, None]
    dist.all_gather_object(hashes, prior.state_sha(aux.state_dict()))
    assert hashes[0] == hashes[1]
    ddp.train()
    opt.zero_grad(set_to_none=True)
    distributed_backward_aux(ddp, x, t, mu, sigma, rank, greens[rank], weight)
    b.finite_grad(aux)
    torch.nn.utils.clip_grad_norm_(aux.parameters(), m['recipe']['gradient_clip'], error_if_nonfinite=True)
    opt.step()
    hashes = [None, None]
    dist.all_gather_object(hashes, prior.state_sha(aux.state_dict()))
    assert hashes[0] == hashes[1]
    if rank == 0:
        report = dict(status='passed', manifest_sha=digest(ROOT/'manifest.json'),
                      aux_manifest_sha=digest(ROOT/'aux_manifest.json'), details=details,
                      single_gpu_reference_gradient_check=evidence[0], ranks_synchronized=True,
                      all_12_encoder_blocks_updated=True, patch_gradient_verified=True, head_updated=True,
                      point_head_gradient_max=point_grad, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        save(ROOT/'smoke.json', report)
        print(json.dumps(report), flush=True)
    dist.barrier()
    dist.destroy_process_group()


def train():
    m, a = verify()
    rank = R.init_dist()
    run = ROOT/'runs'/f'fold{m["fold"]}'/R.ARCH
    for name in ('cpu_tests.json', 'aux_cpu_tests.json', 'smoke.json'):
        check = read(ROOT/name)
        assert check['status'] == 'passed' and check['manifest_sha'] == digest(ROOT/'manifest.json')
    assert read(ROOT/'smoke.json')['aux_manifest_sha'] == digest(ROOT/'aux_manifest.json')
    if rank == 0:
        if run.exists():
            raise FileExistsError('No overwrite or implicit resume')
        run.mkdir(parents=True)
    dist.barrier()
    aux, opt, mu, sigma, details = setup_aux(m, a)
    net = aux.net
    weight = a['aux_weight']
    ddp = DDP(aux, device_ids=[rank], broadcast_buffers=False, find_unused_parameters=True)
    ids, y, tr, va = R.arrays()
    updates = math.ceil(len(tr)/16)
    step = 0
    history = []
    wall = time.monotonic()
    names, gtr, gva = green_split(ids, tr, va)
    ann = read(ANN)
    images = green_images(gtr, y, len(ids))
    targets, support = green_targets(names['train'], ann)
    gtarget = torch.from_numpy(targets).cuda()
    gsupport = torch.from_numpy(support).cuda()
    val = R.loader(va, y, len(ids)) if rank == 0 else None
    probe_ix = np.sort(np.random.default_rng(12012).choice(tr, min(256, len(tr)), replace=False))
    probe_loader = R.loader(probe_ix, y, len(ids)) if rank == 0 else None
    initial_encoder = prior.state_sha(net[0].original.state_dict())
    if rank == 0:
        save(run/'progress.json', dict(status='starting', epochs=m['recipe']['epochs'], world_size=2,
                                      details=details, aux_weight=weight))
    for epoch in range(m['recipe']['epochs']):
        order = np.random.default_rng(m['seed']+epoch).permutation(tr)
        dl = R.loader(order, y, len(ids), epoch, True, batch_size=16)
        seen = []
        tick = time.monotonic()
        permutation = green_permutation(m['seed'], epoch)
        used = np.zeros(20, dtype=np.int64)
        ddp.train()
        torch.cuda.reset_peak_memory_stats()
        totals = torch.zeros(8, device=mu.device, dtype=torch.float64)
        norm_sum = torch.zeros((), device=mu.device)
        norm_max = torch.zeros_like(norm_sum)
        clipped = torch.zeros_like(norm_sum)
        for x, t, ix in dl:
            ratio = R.lr_ratio(step, updates, m['recipe']['schedule_epochs'])
            opt.param_groups[0]['lr'] = m['recipe']['encoder_lr']*ratio
            opt.param_groups[1]['lr'] = m['recipe']['head_lr']*ratio
            opt.param_groups[2]['lr'] = m['recipe']['head_lr']*ratio
            image, flip = green_choice(permutation, m['seed'], epoch, step-epoch*updates, rank)
            used[image] += 1
            green = (flip_image(images[image:image+1], flip), gtarget[image, flip][None], gsupport[image, flip][None])
            opt.zero_grad(set_to_none=True)
            totals += distributed_backward_aux(ddp, x, t, mu, sigma, rank, green, weight)
            norm = torch.nn.utils.clip_grad_norm_(aux.parameters(), m['recipe']['gradient_clip'], error_if_nonfinite=True)
            norm_sum += norm.detach()
            norm_max = torch.maximum(norm_max, norm.detach())
            clipped += (norm.detach() > m['recipe']['gradient_clip']).float()
            opt.step()
            step += 1
            seen.extend(ix.tolist())
            if rank == 0 and (step % 25 == 0 or step == 1):
                item = dict(status='training', epoch=epoch+1, epochs=m['recipe']['epochs'], seen=len(seen),
                            train_n=len(tr), updates=step, world_size=2, gradient_norm=norm.item(),
                            encoder_lr=opt.param_groups[0]['lr'], head_lr=opt.param_groups[1]['lr'],
                            seconds_this_epoch=time.monotonic()-tick,
                            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
                save(run/'progress.json', item)
                print(json.dumps(item), flush=True)
        np.testing.assert_array_equal(np.array(seen), order)
        assert step == (epoch+1)*updates
        dist.all_reduce(totals)
        assert prior.state_sha(net[0].original.state_dict()) != initial_encoder
        dist.barrier()
        train_seconds = time.monotonic()-tick
        if rank == 0:
            pred = b.evaluate(net, R.ARCH, val, mu, sigma, va)
            score = b.metrics(pred, y[va])
            historical_score = R.secondary_score(pred, y, va)
            with R.preserved_rng(rank):
                probe_pred = b.evaluate(net, R.ARCH, probe_loader, mu, sigma, probe_ix)
                probe_score = b.metrics(probe_pred, y[probe_ix])
            values = (totals/len(tr)).cpu().tolist()
            item = dict(epoch=epoch+1, score=score, train_seconds=train_seconds,
                        historical_subset_score=historical_score,
                        online_train_loss=values[0], online_train_mape=dict(zip(('CD', 'CV', 'HEX'), values[1:4])),
                        online_standardized_log_l1=dict(zip(('CD', 'CV', 'HEX'), values[4:7])),
                        online_point_loss=(totals[7]/(2*updates)).item(), green_presentations_rank0=used.tolist(),
                        train_probe_score=probe_score, gradient_norm_mean=(norm_sum/updates).item(),
                        gradient_norm_max=norm_max.item(), gradient_clip_fraction=(clipped/updates).item(),
                        gradient_clip_threshold=m['recipe']['gradient_clip'],
                        encoder_lr=opt.param_groups[0]['lr'], head_lr=opt.param_groups[1]['lr'],
                        epoch_seconds=time.monotonic()-tick, updates=step, order_sha=R.array_sha(order),
                        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
            history.append(item)
            save(run/'history.json', history)
            torch.save(dict(model=net.state_dict(), point_head=aux.point.state_dict(), epoch=epoch+1,
                            architecture=R.ARCH, manifest_sha=digest(ROOT/'manifest.json'),
                            log_mean=m['log_mean'], log_std=m['log_std'],
                            note=f'Model only; terminal epoch{m["recipe"]["epochs"]} primary; no resume'),
                       run/f'epoch_{epoch+1:02d}.pt')
            np.savez_compressed(run/f'predictions_epoch_{epoch+1:02d}.npz', indices=va, ids=ids[va], target=y[va], pred=pred)
            tmp = run/'last.pt.tmp'
            torch.save(dict(model=net.state_dict(), point_head=aux.point.state_dict(), optimizer=opt.state_dict(),
                            epoch=epoch+1, architecture=R.ARCH, manifest_sha=digest(ROOT/'manifest.json'),
                            log_mean=m['log_mean'], log_std=m['log_std'], history=history,
                            note='No resume supported; not both-rank RNG checkpoint'), tmp)
            tmp.replace(run/'last.pt')
            save(run/'progress.json', dict(status='epoch_complete', **item))
            print(json.dumps(dict(epoch_complete=epoch+1, **item)), flush=True)
        dist.barrier()
    hashes = [None, None]
    dist.all_gather_object(hashes, prior.state_sha(aux.state_dict()))
    assert hashes[0] == hashes[1]
    if rank == 0:
        np.savez_compressed(run/'predictions.npz', indices=va, ids=ids[va], target=y[va], pred=pred)
        copy, _ = R.build_cpu(m['seed'])
        copy = AuxNet(copy, build_point_head(m['seed']))
        copy.net.load_state_dict(net.state_dict())
        copy.point.load_state_dict(aux.point.state_dict())
        copy = copy.cuda()
        prior_value = float(np.mean([targets[i, 0][support[i, 0]].mean() for i in range(len(gtr))]))
        validation_rows, probabilities = green_point_eval(copy, gva, names['validation'], ann, y, len(ids), prior_value)
        train_rows, _ = green_point_eval(copy, gtr, names['train'], ann, y, len(ids), prior_value)
        copy.eval()
        x0 = torch.stack([R.Frames(tr[:4], y, len(ids))[i][0] for i in range(4)])
        t0 = torch.from_numpy(y[tr[:4]])
        shares = gradient_shares(copy, x0, t0, (images[:1], gtarget[0, 0][None], gsupport[0, 0][None]), mu, sigma)
        del copy
        np.savez_compressed(run/'green_validation_probabilities.npz', ids=np.array(names['validation']), probability=probabilities)
        probe_reference = read(PROBE_RESULT)
        green = dict(status='completed', aux_weight=weight, epoch=m['recipe']['epochs'],
                     validation_rows=validation_rows, train_rows=train_rows,
                     validation_mean_f1=float(np.mean([r['point']['f1'] for r in validation_rows])),
                     validation_mean_recall=float(np.mean([r['point']['recall'] for r in validation_rows])),
                     validation_baseline_mean_f1=float(np.mean([r['baseline']['f1'] for r in validation_rows])),
                     train_mean_f1=float(np.mean([r['point']['f1'] for r in train_rows])),
                     frozen_probe_validation_mean_f1=probe_reference['mean_f1'], constant_prior=prior_value,
                     terminal_gradient_shares=shares,
                     probabilities_sha=digest(run/'green_validation_probabilities.npz'),
                     note='aux_weight0 point head is untrained (weight decay only); its point metrics are not a probe result')
        save(run/'green_result.json', green)
        result = dict(architecture=R.ARCH, fold=m['fold'], seed=m['seed'], score=score, history=history, details=details,
                      historical_subset_score=historical_score, seconds=time.monotonic()-wall,
                      manifest_sha=digest(ROOT/'manifest.json'), aux_manifest_sha=digest(ROOT/'aux_manifest.json'),
                      aux_weight=weight, checkpoint_sha=digest(run/'last.pt'), predictions_sha=digest(run/'predictions.npz'),
                      green_result_sha=digest(run/'green_result.json'), ranks_synchronized=True, limits=m['limits'],
                      official_submission_changed=False, primary=f'fixed terminal epoch{m["recipe"]["epochs"]}',
                      single_model=True)
        save(run/'result.json', result)
        save(run/'complete.json', dict(status='completed', result_sha=digest(run/'result.json')))
        save(run/'progress.json', dict(status='completed', epoch=m['recipe']['epochs'], score=score, updates=step))
    dist.barrier()
    dist.destroy_process_group()


def audit():
    m, a = verify()
    if not (ROOT/'audit.json').exists():
        R.audit()
    if (ROOT/'aux_audit.json').exists():
        raise FileExistsError('Aux audit already exists')
    run = ROOT/'runs'/f'fold{m["fold"]}'/R.ARCH
    result = prior.validate_result(run, digest(ROOT/'manifest.json'))
    assert result['aux_manifest_sha'] == digest(ROOT/'aux_manifest.json') and result['aux_weight'] == a['aux_weight']
    assert digest(run/'green_result.json') == result['green_result_sha']
    green = read(run/'green_result.json')
    assert digest(run/'green_validation_probabilities.npz') == green['probabilities_sha']
    ann = read(ANN)
    with np.load(run/'green_validation_probabilities.npz') as z:
        assert z['ids'].tolist() == a['green_validation_ids']
        for p, row in zip(z['probability'], green['validation_rows']):
            _, support = spatial.point_target(ann[row['id']]['markers_mha'], ann[row['id']]['roi_box_mha'], GRID)
            assert spatial.match_points(probe.peaks(p, support), ann[row['id']]['markers_mha']) == row['point']
    last = torch.load(run/'last.pt', map_location='cpu', weights_only=True)
    final = torch.load(run/f'epoch_{m["recipe"]["epochs"]:02d}.pt', map_location='cpu', weights_only=True)
    assert prior.state_sha(last['point_head']) == prior.state_sha(final['point_head'])
    regression = read(ROOT/'audit.json')
    report = dict(status='green_aux_regression_verified', aux_weight=a['aux_weight'], score=regression['score'],
                  retrospective_best=regression['retrospective_best'], validation_mean_f1=green['validation_mean_f1'],
                  validation_baseline_mean_f1=green['validation_baseline_mean_f1'],
                  frozen_probe_validation_mean_f1=green['frozen_probe_validation_mean_f1'],
                  terminal_gradient_shares=green['terminal_gradient_shares'],
                  point_metrics_recomputed=True, manifest_sha=digest(ROOT/'manifest.json'),
                  aux_manifest_sha=digest(ROOT/'aux_manifest.json'), limits=a['limits'])
    save(ROOT/'aux_audit.json', report)
    print(json.dumps(report), flush=True)


def main():
    global ARGS, ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'cpu-tests', 'smoke', 'train', 'audit'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--pooling', choices=list(pooling.MODES), required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--ssl-epochs', type=int, choices=[5, 10, 20], required=True)
    parser.add_argument('--ssl-root', type=Path, required=True)
    parser.add_argument('--head-lr', type=float, required=True)
    parser.add_argument('--clip', type=float, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--schedule-epochs', type=int, default=20)
    parser.add_argument('--deadline-utc', required=True)
    parser.add_argument('--aux-weight', type=float, required=True)
    ARGS = parser.parse_args()
    R.parse_deadline(ARGS.deadline_utc)
    R.ARGS, R.ROOT, R.SPLIT, R.SSL = ARGS, ARGS.root, ARGS.split, ARGS.ssl_root
    ROOT = ARGS.root
    globals()[ARGS.command.replace('-', '_')]()


if __name__ == '__main__':
    main()
