"""Frozen point-probe benchmark across existing fold4 encoders; one GPU, no encoder updates.

Same probe contract as green_point_probe_20261023 (linear 768->1 head, ROI-masked
Gaussian BCE, 200 epochs on train20, terminal head, peaks .5 / NMS16 / match12 on
the 5 fold4-validation green images), applied to the SSL5 encoder and to every
completed fold4 seed46 regression encoder. Adds two baselines: the constant
train prior and a dense lattice (constant probability above threshold), which an
untrained head approximates. Preprocessing is the regression Frames + inputs path.
"""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import phase2_foundation_train_20261002 as b
import phase2_foundation_preflight_20261002 as preflight
import green_point_spatial_20261023 as spatial
import green_point_probe_head_20261023 as probe

ROOT = Path('experiments/green_probe_bench_20261027')
ANN = Path('datasets/clear_ec_official/green_eval/annotations.json')
BINDING = Path('experiments/green_point_probe_20261023/data_binding.json')
E = 'experiments/'
ENCODERS = {
    'ssl5': (E+'phase2_crossfold_20261015/fold4/ssl5/run/last.pt', 'encoder'),
    'regression_mean': (E+'phase2_pooling_20261020/arms/mean/runs/fold4/cpdino_vitb/epoch_20.pt', 'model'),
    'regression_std_recipeA': (E+'phase2_pooling_20261020/arms/std/runs/fold4/cpdino_vitb/epoch_20.pt', 'model'),
    'green_control_w0': (E+'green_aux_g1_20261026/control_w0/runs/fold4/cpdino_vitb/epoch_20.pt', 'model'),
    'green_aux_w0p01': (E+'green_aux_g1_20261026/aux_w0p01/runs/fold4/cpdino_vitb/epoch_20.pt', 'model'),
    'green_aux_w0p1': (E+'green_aux_g1_20261026/aux_w0p1/runs/fold4/cpdino_vitb/epoch_20.pt', 'model'),
}
GRID = (124, 164)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def encoder_state(path, key):
    state = torch.load(path, map_location='cpu', weights_only=True)[key]
    if key == 'model':
        state = {k[len('0.original.'):]: v for k, v in state.items() if k.startswith('0.original.')}
    return state


def summarize(rows):
    return dict(mean_f1=float(np.mean([r['f1'] for r in rows])), mean_recall=float(np.mean([r['recall'] for r in rows])),
                mean_precision=float(np.mean([r['precision'] for r in rows])))


def main():
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(4)
    ann = json.loads(ANN.read_text())
    binding = json.loads(BINDING.read_text())
    train, val = binding['train_ids'], binding['validation_ids']
    with np.load(b.META, allow_pickle=True) as z:
        ids, y = z['ids'].astype(str), z['Y'].astype(np.float32)
    lookup = {n: i for i, n in enumerate(ids)}
    names = train+val
    frames = b.Frames(np.array([lookup[n] for n in names]), y, len(ids))
    images = torch.stack([frames[i][0] for i in range(len(names))])
    targets = {n: spatial.point_target(ann[n]['markers_mha'], ann[n]['roi_box_mha'], GRID) for n in names}
    prior = float(np.mean([t[s].mean() for t, s in (targets[n] for n in train)]))
    lattice_rows, prior_rows = [], []
    for n in val:
        support = targets[n][1]
        pts = ann[n]['markers_mha']
        lattice_rows.append(dict(id=n, **spatial.match_points(probe.peaks(np.ones(GRID), support), pts)))
        prior_rows.append(dict(id=n, **spatial.match_points(probe.peaks(np.full(GRID, prior), support), pts)))
    out = dict(created_utc=now(), contract='green_point_probe_20261023 terminal200, fixed threshold .5',
               baselines=dict(lattice=dict(rows=lattice_rows, **summarize(lattice_rows)),
                              constant_prior=dict(rows=prior_rows, **summarize(prior_rows), prior=prior)),
               encoders={}, inputs={str(ANN): sha(ANN), str(BINDING): sha(BINDING)})
    net, _, _ = preflight.build('cpdino_vitb', False, native_convnext_norm=True)
    encoder = net[0].original.cuda().eval().requires_grad_(False)
    for name, (path, key) in ENCODERS.items():
        encoder.load_state_dict(encoder_state(path, key), strict=True)
        feats = {}
        with torch.no_grad():
            for i, n in enumerate(names):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    f = encoder.forward_features(b.inputs(images[i:i+1], 'cpdino_vitb'))['x_norm_patchtokens']
                assert f.shape == (1, GRID[0]*GRID[1], 768) and torch.isfinite(f).all()
                feats[n] = f[0].reshape(*GRID, 768).float()
        torch.manual_seed(46)
        head = probe.build_head().cuda()
        opt = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.0001)
        for epoch in range(1, 201):
            for i in np.random.default_rng(46+epoch).permutation(len(train)):
                n = train[i]
                t, s = targets[n]
                opt.zero_grad()
                loss = probe.loss(head, feats[n], torch.from_numpy(t).cuda(), torch.from_numpy(s).cuda())
                loss.backward()
                opt.step()
        head.eval()

        def evaluate(group, threshold=.5):
            rows = []
            for n in group:
                with torch.inference_mode():
                    p = head(feats[n]).squeeze(-1).sigmoid().cpu().numpy()
                found = probe.peaks(p, targets[n][1], threshold=threshold)
                pts = ann[n]['markers_mha']
                rows.append(dict(id=n, predicted=len(found), annotated=len(pts),
                                 count_absolute_error=abs(len(found)-len(pts)), **spatial.match_points(found, pts)))
            return rows
        thresholds = [round(x, 2) for x in np.arange(.2, .85, .05)]
        train_f1 = {th: summarize(evaluate(train, th))['mean_f1'] for th in thresholds}
        chosen = max(thresholds, key=lambda th: (train_f1[th], -abs(th-.5)))
        primary = evaluate(val)
        secondary = evaluate(val, chosen)
        out['encoders'][name] = dict(checkpoint=path, checkpoint_sha=sha(path), final_train_loss=float(loss),
                                     validation_fixed_threshold=dict(rows=primary, **summarize(primary)),
                                     train_selected_threshold=chosen,
                                     validation_train_selected_threshold=dict(rows=secondary, **summarize(secondary)),
                                     train_fixed_threshold=summarize(evaluate(train)))
        print(json.dumps(dict(encoder=name, val_f1=out['encoders'][name]['validation_fixed_threshold']['mean_f1'],
                              val_f1_train_threshold=out['encoders'][name]['validation_train_selected_threshold']['mean_f1'],
                              threshold=chosen)), flush=True)
        del feats, head, opt
    out['finished_utc'] = now()
    out['limits'] = ['5 validation images; high variance', 'Encoders other than ssl5 were trained on fold4 train labels only',
                     'aux encoders saw train20 green points during regression; validation5 never',
                     'Points/ROI are not cell masks']
    with (ROOT/'result.json').open('x') as f:
        json.dump(out, f, indent=2)


if __name__ == '__main__':
    main()
