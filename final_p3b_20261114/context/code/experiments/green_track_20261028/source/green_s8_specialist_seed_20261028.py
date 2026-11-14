"""Stride-8 point specialist with a seed argument (same recipe as green_point_specialist_fold_20261027: encoder
lr 1e-5, linear head lr 1e-3, AdamW wd 1e-4, clip 1, bf16, batch 2, four exact flips, ROI-masked Gaussian BCE,
1000 updates, train-green 20 only, fixed threshold 0.35).  Writes maps for the fold's validation rows in the
green_point_features extract format under experiments/green_track_20261028/s8_specialist_s<seed>/.
run --fold F --seed S"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parent))
import green_point_features_20261027 as G
import green_point_spatial_20261023 as spatial
import green_point_probe_head_20261023 as probe
import green_selftrain_20261028 as ST
ROOT = Path('experiments/green_track_20261028')
UPDATES, BATCH, TH = 1000, 2, 0.35


def log(msg, **kw):
    print(json.dumps(dict(utc=G.now(), msg=msg, **kw)), flush=True)


def tokens_of(encoder, x, kind):
    xi = G.b.inputs(x, 'cpdino_vitb')
    if kind == 'last':
        return encoder.forward_features(xi)['x_norm_patchtokens']
    return torch.cat(list(encoder.get_intermediate_layers(xi, n=list(G.LAYERS), norm=True)), dim=-1)


def maps_of(encoder, head, x, kind):
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        tok = tokens_of(encoder, x, kind)
    with torch.no_grad():
        return head(tok.float()).squeeze(-1).sigmoid().reshape(len(x), *G.GRID).cpu().numpy()


def main(a):
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(4)
    out = ROOT/(f's8_specialist_s{a.seed}' if a.features == 'last' else f's8multi_specialist_s{a.seed}')/f'fold{a.fold}'
    out.mkdir(parents=True, exist_ok=True)
    ann, binding, ids, y, tr, va = ST.green(a.fold)
    names = binding['train_ids']
    lookup = {n: i for i, n in enumerate(ids)}
    frames = G.b.Frames(np.array([lookup[n] for n in names]), y, len(ids))
    images = {n: frames[i][0][None] for i, n in enumerate(names)}
    targets = {}
    for n in names:
        for f, (h, v) in enumerate(G.FLIPS):
            pts, roi = G.flip_annotation(ann[n]['markers_mha'], ann[n]['roi_box_mha'], h, v)
            t, s = spatial.point_target(pts, roi, G.GRID)
            targets[(n, f)] = (torch.from_numpy(t).cuda(), torch.from_numpy(s).cuda())
    encoder = G.load_encoder(ST.ENCODERS[a.fold]).train().requires_grad_(True)
    head = G.build_head(768 if a.features == 'last' else 768*len(G.LAYERS), 'linear', seed=a.seed)
    opt = torch.optim.AdamW([{'params': encoder.parameters(), 'lr': 1e-5}, {'params': head.parameters(), 'lr': 1e-3}], weight_decay=1e-4)
    rng = np.random.default_rng(a.seed)
    items = [(n, f) for n in names for f in range(4)]
    losses, t0 = [], time.time()
    for step in range(UPDATES):
        pick = [items[i] for i in rng.choice(len(items), BATCH, replace=False)]
        x = torch.cat([G.flip_image(images[n], f) for n, f in pick])
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            tok = tokens_of(encoder, x, a.features)
        logits = head(tok.float()).squeeze(-1).reshape(len(pick), *G.GRID)
        loss = sum(F.binary_cross_entropy_with_logits(logits[i][targets[k][1]], targets[k][0][targets[k][1]]) for i, k in enumerate(pick))/len(pick)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(encoder.parameters())+list(head.parameters()), 1.0, error_if_nonfinite=True)
        opt.step()
        losses.append(float(loss))
        if step % 200 == 0:
            log('train', step=step, loss=float(loss), seconds=round(time.time()-t0, 1))
    encoder.eval().requires_grad_(False); head.eval()
    ST.maps = (lambda enc, hd, x: maps_of(enc, hd, x, a.features))
    val = ST.f1_rows(encoder, head, binding['validation_ids'], ann, ids, y, TH)
    prob = ST.infer_rows(encoder, head, va, y, len(ids))
    train_green = set(names)
    path = out/(f'fold{a.fold}_specialist_s{a.seed}.npz' if a.features == 'last' else f'fold{a.fold}_multi_s{a.seed}.npz')
    np.savez_compressed(path, indices=va, ids=ids[va], probability=prob, point_labels_seen=np.array([n in train_green for n in ids[va]]))
    G.save(out/'result.json', dict(created_utc=G.now(), fold=a.fold, seed=a.seed, features=a.features, init_encoder=ST.ENCODERS[a.fold], updates=UPDATES, batch=BATCH, threshold=TH,
           validation=val, final_train_loss_last100=float(np.mean(losses[-100:])), maps=str(path), maps_sha=G.sha(path), seconds=round(time.time()-t0, 1)))
    log('done', fold=a.fold, seed=a.seed, val_f1=val['mean_f1'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--features', default='last', choices=['last', 'multi'])
    main(ap.parse_args())
