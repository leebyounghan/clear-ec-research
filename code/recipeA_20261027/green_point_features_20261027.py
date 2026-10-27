"""GREEN steps 1-2: point-detector selection on frozen recipe A tokens, then per-fold probability maps.

select: fold4 recipe A encoder (std seed46 terminal). Eight pre-registered configs =
features {last, multi blocks 5/8/11} x head {linear, mlp} x geometry {none, 4 flips}.
5-fold CV over the 20 train-green images picks config and threshold (NMS16, match12,
ROI support as in the frozen probe); the chosen config is retrained on train20 and
reported once on the 5 validation-green images.
extract: for one fold's recipe A encoder, retrain the chosen config on that encoder's
train20 tokens and write fp16 probability maps for every validation image of the fold.
Encoders are never updated. Stacking features are computed on CPU afterwards.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import phase2_foundation_train_20261002 as b
import phase2_foundation_preflight_20261002 as preflight
import green_point_spatial_20261023 as spatial
import green_point_probe_head_20261023 as probe

ROOT = Path('experiments/green_point_features_20261027')
ANN = Path('datasets/clear_ec_official/green_eval/annotations.json')
BINDING = Path('experiments/green_point_probe_20261023/data_binding.json')
SELECT_ENCODER = Path('experiments/phase2_pooling_20261020/arms/std/runs/fold4/cpdino_vitb/epoch_20.pt')
GRID = (124, 164)
WIDTH, HEIGHT = 1296, 972
FLIPS = ((False, False), (True, False), (False, True), (True, True))
LAYERS = (5, 8, 11)
THRESHOLDS = [round(float(x), 2) for x in np.arange(.2, .851, .05)]
CONFIGS = [dict(features=f, head=h, flips=g) for f in ('last', 'multi') for h in ('linear', 'mlp') for g in (False, True)]
EPOCHS, SEED = 200, 46


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(p, d):
    with Path(p).open('x') as f:
        json.dump(d, f, indent=2)


def name_of(cfg):
    return f"{cfg['features']}_{cfg['head']}_{'flips' if cfg['flips'] else 'noflip'}"


def flip_annotation(points, roi, horizontal, vertical):
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


def load_encoder(path):
    net, _, _ = preflight.build('cpdino_vitb', False, native_convnext_norm=True)
    encoder = net[0].original
    state = torch.load(path, map_location='cpu', weights_only=True)['model']
    state = {k[len('0.original.'):]: v for k, v in state.items() if k.startswith('0.original.')}
    encoder.load_state_dict(state, strict=True)
    return encoder.cuda().eval().requires_grad_(False)


def tokens(encoder, x, kind):
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        xi = b.inputs(x, 'cpdino_vitb')
        if kind == 'last':
            t = encoder.forward_features(xi)['x_norm_patchtokens']
        else:
            t = torch.cat(list(encoder.get_intermediate_layers(xi, n=list(LAYERS), norm=True)), dim=-1)
    assert t.shape[1] == GRID[0]*GRID[1] and torch.isfinite(t).all()
    return t.reshape(len(x), *GRID, -1).float()


def build_head(dim, kind, seed=SEED):
    state = torch.random.get_rng_state()
    try:
        torch.random.default_generator.manual_seed(seed)
        if kind == 'linear':
            head = torch.nn.Linear(dim, 1)
        else:
            head = torch.nn.Sequential(torch.nn.Linear(dim, 256), torch.nn.GELU(), torch.nn.Linear(256, 1))
    finally:
        torch.random.set_rng_state(state)
    return head.cuda()


def green_data(encoder, kinds):
    ann = json.loads(ANN.read_text())
    binding = json.loads(BINDING.read_text())
    with np.load(b.META, allow_pickle=True) as z:
        ids, y = z['ids'].astype(str), z['Y'].astype(np.float32)
    lookup = {n: i for i, n in enumerate(ids)}
    names = binding['train_ids']+binding['validation_ids']
    frames = b.Frames(np.array([lookup[n] for n in names]), y, len(ids))
    feats, targets = {k: {} for k in kinds}, {}
    for i, n in enumerate(names):
        x = frames[i][0][None]
        for f, (h, v) in enumerate(FLIPS):
            if n in binding['validation_ids'] and f:
                continue
            xf = flip_image(x, f)
            for k in kinds:
                feats[k][(n, f)] = tokens(encoder, xf, k)[0]
            pts, roi = flip_annotation(ann[n]['markers_mha'], ann[n]['roi_box_mha'], h, v)
            t, s = spatial.point_target(pts, roi, GRID)
            targets[(n, f)] = (torch.from_numpy(t).cuda(), torch.from_numpy(s).cuda(), s)
    return ann, binding, feats, targets


def train_head(feats, targets, names, cfg):
    dim = next(iter(feats.values())).shape[-1]
    head = build_head(dim, cfg['head'])
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    items = [(n, f) for n in names for f in (range(4) if cfg['flips'] else [0])]
    for epoch in range(1, EPOCHS+1):
        for i in np.random.default_rng(SEED+epoch).permutation(len(items)):
            t, s, _ = targets[items[i]]
            opt.zero_grad(set_to_none=True)
            logits = head(feats[items[i]]).squeeze(-1)
            F.binary_cross_entropy_with_logits(logits[s], t[s]).backward()
            opt.step()
    return head.eval()


def probability(head, feat):
    with torch.no_grad():
        return head(feat).squeeze(-1).sigmoid().cpu().numpy()


def score(head, feats, targets, ann, names, thresholds):
    out = {th: [] for th in thresholds}
    for n in names:
        p = probability(head, feats[(n, 0)])
        support = targets[(n, 0)][2]
        for th in thresholds:
            found = probe.peaks(p, support, threshold=th)
            out[th].append(dict(id=n, predicted=len(found), annotated=len(ann[n]['markers_mha']),
                                **spatial.match_points(found, ann[n]['markers_mha'])))
    return out


def select():
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(4)
    (ROOT/'select').mkdir(parents=True, exist_ok=False)
    encoder = load_encoder(SELECT_ENCODER)
    ann, binding, feats, targets = green_data(encoder, ('last', 'multi'))
    train, val = binding['train_ids'], binding['validation_ids']
    groups = np.array_split(np.random.default_rng(SEED).permutation(len(train)), 5)
    cv = {}
    for cfg in CONFIGS:
        rows = {th: [] for th in THRESHOLDS}
        for g in groups:
            held = [train[i] for i in g]
            fit = [n for n in train if n not in held]
            head = train_head(feats[cfg['features']], targets, fit, cfg)
            for th, r in score(head, feats[cfg['features']], targets, ann, held, THRESHOLDS).items():
                rows[th].extend(r)
        cv[name_of(cfg)] = {str(th): float(np.mean([r['f1'] for r in rows[th]])) for th in THRESHOLDS}
        print(json.dumps(dict(config=name_of(cfg), best_cv_f1=max(cv[name_of(cfg)].values()))), flush=True)
    complexity = {c: i for i, c in enumerate(name_of(c) for c in CONFIGS)}
    best_name, best_th = max(((c, float(th)) for c in cv for th in cv[c]),
                             key=lambda k: (round(cv[k[0]][str(k[1])], 6), -complexity[k[0]], -abs(k[1]-.5)))
    cfg = next(c for c in CONFIGS if name_of(c) == best_name)
    head = train_head(feats[cfg['features']], targets, train, cfg)
    primary = score(head, feats[cfg['features']], targets, ann, val, [best_th])[best_th]
    reference_cfg = dict(features='last', head='linear', flips=False)
    reference = score(train_head(feats['last'], targets, train, reference_cfg), feats['last'], targets, ann, val, [.5])[.5]
    torch.save(dict(head=head.state_dict(), config=cfg, threshold=best_th), ROOT/'select'/'head_fold4.pt')
    result = dict(created_utc=now(), encoder=str(SELECT_ENCODER), encoder_sha=sha(SELECT_ENCODER),
                  configs=[name_of(c) for c in CONFIGS], thresholds=THRESHOLDS, cv_mean_f1=cv,
                  chosen=dict(config=cfg, name=best_name, threshold=best_th, cv_f1=cv[best_name][str(best_th)]),
                  validation_chosen=dict(rows=primary, mean_f1=float(np.mean([r['f1'] for r in primary])),
                                         mean_recall=float(np.mean([r['recall'] for r in primary])),
                                         mean_precision=float(np.mean([r['precision'] for r in primary]))),
                  validation_reference_probe_contract=dict(rows=reference, mean_f1=float(np.mean([r['f1'] for r in reference]))),
                  lattice_baseline_validation_f1=0.378, selection='5-fold CV on train20 only; validation5 opened once for the chosen config and the reference',
                  limits=['20+5 green images; high variance', 'ROI support given at evaluation, as in the frozen probe'])
    save(ROOT/'select'/'result.json', result)
    print(json.dumps(dict(chosen=best_name, threshold=best_th, val_f1=result['validation_chosen']['mean_f1'],
                          reference_val_f1=result['validation_reference_probe_contract']['mean_f1'])), flush=True)


def extract(fold, encoder_path, split, tag):
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(4)
    chosen = json.loads((ROOT/'select'/'result.json').read_text())['chosen']
    cfg, th = chosen['config'], chosen['threshold']
    out_dir = ROOT/'extract'
    out_dir.mkdir(parents=True, exist_ok=True)
    encoder = load_encoder(encoder_path)
    ann, binding, feats, targets = green_data(encoder, (cfg['features'],))
    head = train_head(feats[cfg['features']], targets, binding['train_ids'], cfg)
    green_val = score(head, feats[cfg['features']], targets, ann, binding['validation_ids'], [th])[th]
    del feats
    with np.load(b.META, allow_pickle=True) as z:
        ids, y = z['ids'].astype(str), z['Y'].astype(np.float32)
    with np.load(split) as z:
        va = z['validation'].copy()
    dl = DataLoader(b.Frames(va, y, len(ids)), batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
    maps, seen = [], []
    for x, _, ix in dl:
        t = tokens(encoder, x, cfg['features'])
        with torch.no_grad():
            maps.append(head(t).squeeze(-1).sigmoid().half().cpu().numpy())
        seen.append(ix.numpy())
    seen = np.concatenate(seen)
    np.testing.assert_array_equal(seen, va)
    green_names = set(binding['train_ids'])
    path = out_dir/f'fold{fold}_{tag}.npz'
    np.savez_compressed(path, indices=va, ids=ids[va], probability=np.concatenate(maps),
                        point_labels_seen=np.array([n in green_names for n in ids[va]]))
    save(out_dir/f'fold{fold}_{tag}.json', dict(created_utc=now(), fold=fold, encoder=str(encoder_path), encoder_sha=sha(encoder_path),
         split=str(split), split_sha=sha(split), config=cfg, threshold=th, n=len(va), maps_sha=sha(path),
         green_validation_f1=float(np.mean([r['f1'] for r in green_val])), green_validation_rows=green_val,
         point_labels_seen_in_fold_validation=int(sum(n in green_names for n in ids[va]))))
    print(json.dumps(dict(fold=fold, tag=tag, n=len(va), green_val_f1=float(np.mean([r['f1'] for r in green_val])))), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['select', 'extract'])
    ap.add_argument('--fold', type=int)
    ap.add_argument('--encoder', type=Path)
    ap.add_argument('--split', type=Path)
    ap.add_argument('--tag')
    a = ap.parse_args()
    if a.command == 'select':
        select()
    else:
        extract(a.fold, a.encoder, a.split, a.tag)
