"""Matched raw-target ablation. Fresh heads; fold-specific base/SSL encoder."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time
import numpy as np
import cv2
import torch
import torch.utils.checkpoint as checkpoint
from prepare_target_screen import ROOT, PRIOR, sha
from target_screen_core import targets_for_training

sys.path.insert(0,'experiments/crop_dino_screen_20260910/source/frozen')
import train_specialize_screen as shared


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--fold',type=int,choices=(0,1),required=True)
    parser.add_argument('--initialization',choices=('base','ssl'),required=True)
    parser.add_argument('--policy',choices=('raw','slidesmooth'),default='raw')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.policy!='raw' and not args.smoke:
        raise ValueError('Full smoothed controls already exist; do not duplicate them')
    manifest=json.loads((ROOT/'manifest.json').read_text())
    for p,h in manifest['source_sha256'].items():assert sha(p)==h,p
    assert sha('cpsam_models/cpsam_v2')==manifest['initial_sha256']
    fold=manifest['folds'][str(args.fold)]
    control=fold['controls'][args.initialization]
    old=Path(control['folder'])
    assert sha(old/'config.json')==control['config_sha256']
    assert sha(old/'selected.pt')==control['checkpoint_sha256']
    assert sha(fold['split_path'])==fold['split_sha256']
    out=ROOT/('smoke' if args.smoke else 'runs')/f'{args.initialization}_{args.policy}_fold{args.fold}'
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    torch.set_float32_matmul_precision('high');shared.seed(shared.SEED)
    historic=out/'historic_input';historic.mkdir()
    ids,y,old_tr,va,old_z,views,_=shared.load_data(args.fold,historic)
    del old_z
    with np.load(fold['split_path']) as d:
        tr=d['train'].copy()
        np.testing.assert_array_equal(va,d['validation'])
        np.testing.assert_array_equal(old_tr,d['historical_train'])
    with Path('datasets/clear_ec_official/final_train_ids.csv').open() as f:
        mapping={shared.clean_id(r['ID']):str(r['slide_id']) for r in csv.DictReader(f)}
    slides=np.array([mapping[shared.clean_id(i)] for i in ids])
    assert not set(slides[tr])&set(slides[va])
    targets=targets_for_training(y,slides,tr,args.policy)
    np.testing.assert_array_equal(targets[va],y[va])
    z=torch.tensor(((np.log(targets.clip(1e-6))-shared.LM)/shared.LS).astype(np.float32),device='cuda')
    # Preserve exact historical RNG sequence: same fresh base construction,
    # then overwrite encoder only for the matching fold-specific SSL branch.
    shared.seed(shared.SEED)
    net=shared.build_model('cpsam_models/cpsam_v2')
    if args.initialization=='ssl':
        assert sha(control['encoder_path'])==control['encoder_sha256']
        ck=torch.load(control['encoder_path'],map_location='cpu',weights_only=True)
        assert ck['step']==6000 and ck['config']['validation_images_used']==0
        net.encoder.load_state_dict(ck['teacher_encoder'],strict=True);del ck
    for block in net.encoder.blocks:
        original=block.forward
        block.forward=lambda x,fn=original:checkpoint.checkpoint(fn,x,use_reentrant=False)
    updates=2 if args.smoke else 35*(len(tr)//48)
    config=dict(initialization=args.initialization,target_policy=args.policy,fold=args.fold,
        smoke=args.smoke,seed=77,updates=updates,epochs_equivalent=35,k_crops=2,batch_size=48,
        encoder_lr=1e-5,head_lr=1e-3,objective='mixed_mape',region_policy='full_quality',
        select_mode='fixed_terminal_no_validation_selection',initial_sha256=manifest['initial_sha256'],
        ssl_checkpoint_sha256=control.get('encoder_sha256'),split_sha256=fold['split_sha256'],
        train_rows=len(tr),validation_rows=len(va),lm=shared.LM.tolist(),ls=shared.LS.tolist(),
        historical_validation_reuse=True,matched_control=str(old),only_changed_factor='target_policy')
    shared.save_json(out/'config.json',config)
    np.savez_compressed(out/'split_indices.npz',train=tr,validation=va)
    training=shared.train(net,views,tr,z,None,updates,out,1e-5,77)
    if not args.smoke:
        assert training['first_three_batches_sha256']==control['sampling_sha256'],'Input sampling changed'
    torch.save(dict(model=net.state_dict(),config=config),out/'selected.pt')
    ck=torch.load(out/'selected.pt',map_location='cpu',weights_only=True)
    net.load_state_dict(ck['model'],strict=True);del ck
    evaluation=shared.evaluate(net,views,va[:4] if args.smoke else va,y,ids,out)
    control_parity=None
    if args.smoke and args.policy=='slidesmooth' and args.initialization=='base':
        # Old smoke is exactly the same original initializer. SSL smoke had a
        # four-step encoder, so it is NOT a valid numerical parity reference here.
        with np.load(PRIOR/'smoke'/f'supervised55_fold{args.fold}'/'oof.npz') as a,np.load(out/'oof.npz') as b:
            for key in ('indices','ids','target'):np.testing.assert_array_equal(a[key],b[key])
            np.testing.assert_allclose(a['pred'],b['pred'],rtol=1e-5,atol=1e-6)
        control_parity=True
    result=dict(completed_at=time.strftime('%FT%T%z'),training=training,evaluation=evaluation,
        checkpoint_sha256=sha(out/'selected.pt'),strict_reload_passed=True,smoke=args.smoke,
        smoothed_base_smoke_parity=control_parity,deployment_approved=False)
    shared.save_json(out/'result.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
