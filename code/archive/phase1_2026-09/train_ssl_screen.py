"""One GPU, one fold, fresh CPSAM; no retry or checkpoint selection on validation."""
import argparse
import csv
import gc
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import torch
import torch.utils.checkpoint as checkpoint

ROOT = Path('experiments/ssl_screen_20260911')
sys.path.insert(0, 'experiments/crop_dino_screen_20260910/source/frozen')
import train_specialize_screen as shared
sys.path.insert(0, str(ROOT/'source'))
from ssl_native_core import pretrain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, choices=(0, 1), required=True)
    parser.add_argument('--recipe', choices=('ssl35', 'supervised55'), required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((ROOT/'manifest.json').read_text())
    for path, expected in manifest['source_sha256'].items():
        assert shared.digest(path) == expected, path
    assert shared.digest('cpsam_models/cpsam_v2') == manifest['initial_sha256']
    out = ROOT/('smoke' if args.smoke else 'runs')/f'{args.recipe}_fold{args.fold}'
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4); cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    shared.seed(shared.SEED)
    historic_input = out/'historic_input'; historic_input.mkdir()
    ids, y, old_tr, va, z, views, old = shared.load_data(args.fold, historic_input)
    fold_record = manifest['folds'][str(args.fold)]
    assert shared.digest(old/'split_indices.npz') == fold_record['historical_split_sha256']
    assert shared.digest(ROOT/'preflight'/f'fold{args.fold}.npz') == fold_record['split_sha256']
    with np.load(ROOT/'preflight'/f'fold{args.fold}.npz') as split:
        tr = split['train'].copy()
        np.testing.assert_array_equal(old_tr, split['historical_train'])
        np.testing.assert_array_equal(va, split['validation'])
    # Recompute smoothed labels after removing duplicate-connected training groups.
    # The historical helper's old smoothed targets must not enter either loss.
    with Path('datasets/clear_ec_official/final_train_ids.csv').open() as stream:
        mapping={shared.clean_id(r['ID']):str(r['slide_id']) for r in csv.DictReader(stream)}
    slides=np.asarray([mapping[shared.clean_id(i)] for i in ids])
    targets=y.copy()
    for slide in np.unique(slides[tr]):
        ix=tr[slides[tr]==slide]
        if len(ix)>1:targets[ix]=y[ix].mean(0)
    z=torch.tensor(((np.log(targets.clip(1e-6))-shared.LM)/shared.LS).astype(np.float32),device='cuda')
    np.savez_compressed(out/'split_indices.npz',train=tr,validation=va)
    ssl = None
    if args.recipe == 'ssl35':
        ssl_out = out/'pretrain'; ssl_out.mkdir()
        net = shared.build_model('cpsam_models/cpsam_v2')
        ssl = pretrain(net.encoder, views, tr, ssl_out, 4 if args.smoke else 6000, shared)
        del net; gc.collect(); torch.cuda.empty_cache()
    # The same head initialization and downstream RNG as the historical control.
    shared.seed(shared.SEED)
    net = shared.build_model('cpsam_models/cpsam_v2')
    if ssl:
        state = torch.load(ssl['final_checkpoint'], map_location='cpu', weights_only=True)
        net.encoder.load_state_dict(state['teacher_encoder'], strict=True)
        del state
    for block in net.encoder.blocks:
        original = block.forward
        block.forward = lambda x, fn=original: checkpoint.checkpoint(fn, x, use_reentrant=False)
    epochs = 35
    updates = 2 if args.smoke else epochs*(len(tr)//shared.BS)
    config = dict(recipe=args.recipe, fold=args.fold, smoke=args.smoke,
                  seed=shared.SEED, updates=updates, epochs_equivalent=epochs,
                  k_crops=shared.K, batch_size=shared.BS, encoder_lr=1e-5, head_lr=1e-3,
                  objective='mixed_mape', select_mode='fixed_terminal_no_validation_selection',
                  lm=shared.LM.tolist(), ls=shared.LS.tolist(), region_policy='full_quality',
                  initial_sha256=manifest['initial_sha256'],
                  ssl_checkpoint_sha256=ssl['checkpoint_sha256'] if ssl else None,
                  train_rows=len(tr), validation_rows=len(va),
                  split_sha256=fold_record['split_sha256'], historic_validation_reuse=True)
    shared.save_json(out/'config.json', config)
    training = shared.train(net, views, tr, z, None, updates, out, 1e-5, shared.SEED)
    # Sampling parity is compared against the NEW clean35 control by the audit.
    torch.save(dict(model=net.state_dict(), config=config), out/'selected.pt')
    state = torch.load(out/'selected.pt', map_location='cpu', weights_only=True)
    net.load_state_dict(state['model'], strict=True); del state
    evaluation = shared.evaluate(net, views, va[:4] if args.smoke else va, y, ids, out)
    result = dict(completed_at=time.strftime('%FT%T%z'), pretraining=ssl, training=training,
                  evaluation=evaluation, checkpoint_sha256=shared.digest(out/'selected.pt'),
                  strict_reload_passed=True, deployment_approved=False, smoke=args.smoke)
    shared.save_json(out/'result.json', result)
    if args.recipe == 'supervised55':
        # Preserve the clean35 comparator. Continue this same model for the
        # extra-compute control, with a predeclared lower LR and fresh optimizer.
        extra=out/'continued55'; extra.mkdir()
        extra_updates=2 if args.smoke else 20*(len(tr)//shared.BS)
        extra_config=dict(config,recipe='continued55',epochs_equivalent=55,
                          updates=updates+extra_updates,continuation_updates=extra_updates,
                          continuation_encoder_lr=3e-6,continuation_head_lr=3e-4,
                          continuation_optimizer_reset=True,continuation_seed=shared.SEED+100,
                          initial_clean35_checkpoint_sha256=result['checkpoint_sha256'])
        shared.save_json(extra/'config.json',extra_config)
        np.savez_compressed(extra/'split_indices.npz',train=tr,validation=va)
        extra_training=shared.train(net,views,tr,z,None,extra_updates,extra,3e-6,shared.SEED+100)
        torch.save(dict(model=net.state_dict(),config=extra_config),extra/'selected.pt')
        state=torch.load(extra/'selected.pt',map_location='cpu',weights_only=True)
        net.load_state_dict(state['model'],strict=True);del state
        extra_eval=shared.evaluate(net,views,va[:4] if args.smoke else va,y,ids,extra)
        shared.save_json(extra/'result.json',dict(completed_at=time.strftime('%FT%T%z'),
            pretraining=None,training=extra_training,evaluation=extra_eval,
            initial_training=training,initial_evaluation=evaluation,
            checkpoint_sha256=shared.digest(extra/'selected.pt'),strict_reload_passed=True,
            deployment_approved=False,smoke=args.smoke))
        shared.save_json(out/'worker_complete.json',dict(continued55_complete=True))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
