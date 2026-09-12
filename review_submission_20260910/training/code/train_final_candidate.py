"""Finite all-data retraining, OOF-median optimizer-step selection, no holdout tuning."""
import argparse
import fcntl
import hashlib
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.utils.checkpoint as checkpoint
from final_candidate_core import LM,LS,build_model,train_view,save_json

ROOT=Path('experiments/final_candidate_20260904')
DATA=Path('datasets/clear_ec_final9000_20260904')


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--smoke',action='store_true'); args=parser.parse_args()
    assert args.seed in [42,77,2026]
    run=ROOT/('smoke' if args.smoke else 'runs')/f'seed{args.seed}'
    run.mkdir(parents=True,exist_ok=True)
    with (run/'train.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (run/'result.json').exists(): print('already complete'); return
        if (run/'config.json').exists(): raise RuntimeError('Interrupted run: explicitly inspect before restart')
        plan=json.loads((ROOT/'plan.json').read_text())
        target=int(plan['selected_steps'][str(args.seed)])
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
        torch.set_float32_matmul_precision('high')
        meta=np.load(DATA/'meta.npz'); indices=meta['train_indices']
        y=meta['Y_train']; z=torch.tensor(((np.log(np.clip(y,1e-6,None))-LM)/LS).astype(np.float32),device='cuda')
        imgs=np.memmap(DATA/'img_native_512.dat',dtype='uint8',mode='r',shape=(9000,512,512))
        net=build_model('cpsam_models/cpsam_v2')
        for block in net.encoder.blocks:
            block._candidate_forward=block.forward
            block.forward=(lambda m: lambda x: checkpoint.checkpoint(m._candidate_forward,x,use_reentrant=False))(block)
        heads=list(net.norm.parameters())+list(net.cd.parameters())+list(net.cv.parameters())+list(net.hex.parameters())
        opt=torch.optim.AdamW([dict(params=net.encoder.parameters(),lr=1e-5),dict(params=heads,lr=1e-3)],weight_decay=1e-4)
        covered=[id(p) for g in opt.param_groups for p in g['params']]
        assert len(covered)==len(set(covered)) and set(covered)=={id(p) for p in net.parameters() if p.requires_grad}
        scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda step:min(1.,(step+1)/200))
        bs=4 if args.smoke else 48; epochs=1 if args.smoke else 25
        config=dict(seed=args.seed,epochs=epochs,batch_size=bs,pool='mean',k_crops=1,lm=LM.tolist(),ls=LS.tolist(),
            optimizer_version='all_trainable_v2',selected_steps=target,selection='median OOF best optimizer updates',
            train_rows=len(indices),validation_rows=0,final_epoch_is_secondary_not_selected=True,
            last_batch='Include partial batch so all eligible images participate every epoch',
            source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                ['code/train_final_candidate.py','code/final_candidate_core.py']})
        save_json(run/'config.json',config)
        def save_checkpoint(name,epoch,step):
            dest=run/name; temp=dest.with_suffix('.partial.pt')
            torch.save(dict(model=net.state_dict(),config=config,epoch=epoch,step=step),temp); temp.rename(dest)
        step=0; start=time.time(); selected=False
        for ep in range(epochs):
            net.train(); epoch_start=time.time(); perm=np.random.default_rng(args.seed*1000+ep).permutation(indices)
            batches=1 if args.smoke else math.ceil(len(perm)/bs); total=0
            for k in range(batches):
                ix=perm[k*bs:(k+1)*bs]
                x=torch.from_numpy(np.stack([train_view(imgs[i]) for i in ix])).to('cuda')
                with torch.autocast('cuda',dtype=torch.bfloat16): loss=(net(x)-z[ix]).abs().mean()
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(),1.); opt.step(); scheduler.step()
                step+=1; total+=loss.item()
                if step==target or args.smoke:
                    save_checkpoint('selected.pt',ep+1,step); selected=True
                if step%20==0:
                    save_json(run/'progress.json',dict(epoch=ep+1,step=step,seconds=time.time()-start,
                        updated_unix=time.time(),loss=loss.item()))
            print(f'epoch={ep+1:02d} steps={step} loss={total/batches:.6f} seconds={time.time()-epoch_start:.1f}',flush=True)
            if (ep+1)%5==0 or args.smoke: save_checkpoint(f'epoch{ep+1:02d}.pt',ep+1,step)
        assert selected
        save_json(run/'result.json',dict(seed=args.seed,selected=str(run/'selected.pt'),selected_steps=target,
            completed_steps=step,elapsed_seconds=time.time()-start,completed_at=time.strftime('%FT%T%z'),
            note='No validation score. Selection fixed before all-data training.'))


if __name__=='__main__': main()
