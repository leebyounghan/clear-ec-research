"""Point specialist for one fold: fine-tune a copy of that fold's recipe A encoder on train20, write maps.

Same recipe as green_point_specialist_20261027 (encoder lr 1e-5, linear head lr 1e-3, 1000 updates,
batch 2, four exact flips, ROI-masked Gaussian BCE). The threshold is the fold4-selected 0.35 and is
not re-selected. The fold's regression model is untouched; maps cover the fold's validation set.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import green_point_features_20261027 as G
import green_point_specialist_20261027 as S


def main(fold, encoder, split, tag, threshold):
    assert torch.cuda.device_count() == 1
    torch.set_num_threads(4)
    out = G.ROOT/'specialist'/f'fold{fold}_{tag}'
    out.mkdir(parents=True, exist_ok=False)
    G.SELECT_ENCODER = encoder
    ann, binding, ids, y, images, targets = S.data()
    train, val = binding['train_ids'], binding['validation_ids']
    enc, head, losses = S.finetune(train, images, targets)
    val_rows = S.rows_for(enc, head, val, images, targets, ann, [threshold])[threshold]
    torch.save(dict(encoder=enc.state_dict(), head=head.state_dict(), threshold=threshold, init_encoder=str(encoder)), out/'specialist.pt')
    with np.load(split) as z:
        va = z['validation'].copy()
    dl = DataLoader(G.b.Frames(va, y, len(ids)), batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
    probs, seen = [], []
    for x, _, ix in dl:
        probs.append(S.maps(enc, head, x).astype(np.float16))
        seen.append(ix.numpy())
    np.testing.assert_array_equal(np.concatenate(seen), va)
    path = G.ROOT/'extract'/f'fold{fold}_specialist_{tag}.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    green_names = set(train)
    np.savez_compressed(path, indices=va, ids=ids[va], probability=np.concatenate(probs),
                        point_labels_seen=np.array([n in green_names for n in ids[va]]))
    G.save(out/'result.json', dict(created_utc=G.now(), fold=fold, init_encoder=str(encoder), init_sha=G.sha(encoder), split=str(split),
           split_sha=G.sha(split), updates=S.UPDATES, batch=S.BATCH, threshold=threshold, threshold_origin='fold4 CV selection, not re-selected',
           validation=dict(rows=val_rows, mean_f1=float(np.mean([r['f1'] for r in val_rows])),
                           mean_recall=float(np.mean([r['recall'] for r in val_rows])), mean_precision=float(np.mean([r['precision'] for r in val_rows]))),
           final_train_loss_last100=float(np.mean(losses[-100:])), maps=str(path), maps_sha=G.sha(path), n=len(va),
           point_labels_seen_in_fold_validation=int(sum(n in green_names for n in ids[va])),
           note='validation-green5 metrics are diagnostic; for folds 0-3 the green images inside the fold validation set are flagged for exclusion from stacking'))
    print(json.dumps(dict(fold=fold, tag=tag, val_f1=float(np.mean([r['f1'] for r in val_rows])), n=len(va))), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--encoder', type=Path, required=True)
    ap.add_argument('--split', type=Path, required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--threshold', type=float, default=0.35)
    a = ap.parse_args()
    main(a.fold, a.encoder, a.split, a.tag, a.threshold)
