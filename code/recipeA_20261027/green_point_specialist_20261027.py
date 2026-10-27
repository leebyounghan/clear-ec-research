"""GREEN point specialist: fine-tune a copy of the fold4 recipe A encoder for points only; one GPU.

The regression model is untouched; the specialist exists only to produce point maps
for stacking. Encoder lr 1e-5, linear 768->1 head lr 1e-3, AdamW wd 1e-4, global clip
1, bf16, batch 2, exact 4-flip geometry, ROI-masked Gaussian BCE, fixed 1000 updates.
The threshold is chosen by 5-fold CV fine-tunes on train20; the final specialist is
trained on train20, reported once on validation-green5, then writes fold4 maps.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import green_point_features_20261027 as G

OUT = G.ROOT/'specialist'
UPDATES, BATCH = 1000, 2
FOLD4_SPLIT = Path('experiments/target_fold4_20260912/preflight/fold4.npz')


def data():
    ann = json.loads(G.ANN.read_text())
    binding = json.loads(G.BINDING.read_text())
    with np.load(G.b.META, allow_pickle=True) as z:
        ids, y = z['ids'].astype(str), z['Y'].astype(np.float32)
    lookup = {n: i for i, n in enumerate(ids)}
    names = binding['train_ids']+binding['validation_ids']
    frames = G.b.Frames(np.array([lookup[n] for n in names]), y, len(ids))
    images = {n: frames[i][0][None] for i, n in enumerate(names)}
    targets = {}
    for n in names:
        for f, (h, v) in enumerate(G.FLIPS):
            pts, roi = G.flip_annotation(ann[n]['markers_mha'], ann[n]['roi_box_mha'], h, v)
            t, s = G.spatial.point_target(pts, roi, G.GRID)
            targets[(n, f)] = (torch.from_numpy(t).cuda(), torch.from_numpy(s).cuda(), s)
    return ann, binding, ids, y, images, targets


def finetune(names, images, targets):
    encoder = G.load_encoder(G.SELECT_ENCODER).train().requires_grad_(True)
    head = G.build_head(768, 'linear')
    opt = torch.optim.AdamW([{'params': encoder.parameters(), 'lr': 1e-5}, {'params': head.parameters(), 'lr': 1e-3}],
                            weight_decay=1e-4)
    rng = np.random.default_rng(G.SEED)
    items = [(n, f) for n in names for f in range(4)]
    losses = []
    for step in range(UPDATES):
        pick = [items[i] for i in rng.choice(len(items), BATCH, replace=False)]
        x = torch.cat([G.flip_image(images[n], f) for n, f in pick])
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            tok = encoder.forward_features(G.b.inputs(x, 'cpdino_vitb'))['x_norm_patchtokens']
        logits = head(tok.float()).squeeze(-1).reshape(len(pick), *G.GRID)
        loss = sum(F.binary_cross_entropy_with_logits(logits[i][targets[k][1]], targets[k][0][targets[k][1]])
                   for i, k in enumerate(pick))/len(pick)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(encoder.parameters())+list(head.parameters()), 1.0, error_if_nonfinite=True)
        opt.step()
        losses.append(float(loss))
    return encoder.eval().requires_grad_(False), head.eval(), losses


def maps(encoder, head, x):
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        tok = encoder.forward_features(G.b.inputs(x, 'cpdino_vitb'))['x_norm_patchtokens']
    with torch.no_grad():
        return head(tok.float()).squeeze(-1).sigmoid().reshape(len(x), *G.GRID).cpu().numpy()


def rows_for(encoder, head, names, images, targets, ann, thresholds):
    out = {th: [] for th in thresholds}
    for n in names:
        p = maps(encoder, head, images[n])[0]
        for th in thresholds:
            found = G.probe.peaks(p, targets[(n, 0)][2], threshold=th)
            out[th].append(dict(id=n, predicted=len(found), annotated=len(ann[n]['markers_mha']),
                                **G.spatial.match_points(found, ann[n]['markers_mha'])))
    return out


def main():
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(4)
    OUT.mkdir(parents=True, exist_ok=False)
    ann, binding, ids, y, images, targets = data()
    train, val = binding['train_ids'], binding['validation_ids']
    groups = np.array_split(np.random.default_rng(G.SEED).permutation(len(train)), 5)
    cv = {th: [] for th in G.THRESHOLDS}
    for g in groups:
        held = [train[i] for i in g]
        encoder, head, _ = finetune([n for n in train if n not in held], images, targets)
        for th, r in rows_for(encoder, head, held, images, targets, ann, G.THRESHOLDS).items():
            cv[th].extend(r)
        del encoder, head
        torch.cuda.empty_cache()
    cv_f1 = {str(th): float(np.mean([r['f1'] for r in cv[th]])) for th in G.THRESHOLDS}
    th = max(G.THRESHOLDS, key=lambda t: (round(cv_f1[str(t)], 6), -abs(t-.5)))
    encoder, head, losses = finetune(train, images, targets)
    val_rows = rows_for(encoder, head, val, images, targets, ann, [th])[th]
    torch.save(dict(encoder=encoder.state_dict(), head=head.state_dict(), threshold=th), OUT/'specialist_fold4.pt')
    with np.load(FOLD4_SPLIT) as z:
        va = z['validation'].copy()
    dl = DataLoader(G.b.Frames(va, y, len(ids)), batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
    probs, seen = [], []
    for x, _, ix in dl:
        probs.append(maps(encoder, head, x).astype(np.float16))
        seen.append(ix.numpy())
    np.testing.assert_array_equal(np.concatenate(seen), va)
    path = G.ROOT/'extract'/'fold4_specialist_s46.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    green_names = set(train)
    np.savez_compressed(path, indices=va, ids=ids[va], probability=np.concatenate(probs),
                        point_labels_seen=np.array([n in green_names for n in ids[va]]))
    G.save(OUT/'result.json', dict(created_utc=G.now(), init_encoder=str(G.SELECT_ENCODER), init_sha=G.sha(G.SELECT_ENCODER),
           updates=UPDATES, batch=BATCH, cv_mean_f1=cv_f1, threshold=th,
           validation=dict(rows=val_rows, mean_f1=float(np.mean([r['f1'] for r in val_rows])),
                           mean_recall=float(np.mean([r['recall'] for r in val_rows])),
                           mean_precision=float(np.mean([r['precision'] for r in val_rows]))),
           final_train_loss_last100=float(np.mean(losses[-100:])), maps=str(path), maps_sha=G.sha(path),
           reference=dict(frozen_recipeA_probe=0.556, lattice=0.378),
           limits=['20 training / 5 validation green images', 'Fixed 1000 updates; no validation-based stopping']))
    print(json.dumps(dict(threshold=th, val_f1=float(np.mean([r['f1'] for r in val_rows])))), flush=True)


if __name__ == '__main__':
    main()
