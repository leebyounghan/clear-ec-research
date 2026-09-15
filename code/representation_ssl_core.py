"""Two controlled modifications of the frozen CLEAR-EC masked EMA SSL.

Not an implementation of DINOv3 or VICReg. No new external weights or labels.
"""
import copy
import math
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.checkpoint as checkpoint
from ssl_native_core import normalized_tokens, penalties, feature_diagnostics, require_healthy, mask_batch

VARIANTS = {'low_cov': {'covariance_weight': .001, 'anchor_weight': 0.},
            'spatial_anchor': {'covariance_weight': .01, 'anchor_weight': .1}}


def spatial_relation_loss(features, reference, mask):
    """Preserve cosine relationships at visible positions, not raw feature values."""
    a = F.normalize(normalized_tokens(features)[:, ::4, ::4].flatten(1, 2), dim=-1)
    b = F.normalize(normalized_tokens(reference.detach())[:, ::4, ::4].flatten(1, 2), dim=-1)
    visible = (1-mask[:, 0, ::4, ::4]).flatten(1)
    weights = visible[:, :, None] * visible[:, None, :]
    weights = weights * (1-torch.eye(a.shape[1], device=a.device)[None])
    return (((a @ a.transpose(1, 2) - b @ b.transpose(1, 2))**2)*weights).sum()/weights.sum().clamp_min(1.)


def pretrain(encoder, views, train_indices, destination, steps, shared, variant):
    settings = VARIANTS[variant]
    shared.seed(20260911)
    teacher = copy.deepcopy(encoder).eval()
    anchor = copy.deepcopy(encoder).eval() if settings['anchor_weight'] else None
    for frozen in (teacher, anchor):
        if frozen is not None:
            for p in frozen.parameters(): p.requires_grad_(False)
    predictor = nn.Sequential(nn.Conv2d(256,512,3,padding=1), nn.GroupNorm(32,512),
                              nn.GELU(), nn.Conv2d(512,256,3,padding=1)).cuda()
    for block in encoder.blocks:
        original = block.forward
        block.forward = lambda x, fn=original: checkpoint.checkpoint(fn,x,use_reentrant=False)
    encoder.train()
    parameters = list(encoder.parameters())+list(predictor.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=1e-5, weight_decay=.05)
    permitted = set(map(int,train_indices)); batch=32
    probes = np.random.default_rng(910).choice(train_indices,size=32,replace=False)
    probe = np.stack([views.tile(int(i),views.selected[i,0],0) for i in probes])
    @torch.no_grad()
    def diagnose():
        feats=[]
        for i in range(0,len(probe),8):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                feats.append(teacher(torch.from_numpy(probe[i:i+8]).cuda()).float())
        return feature_diagnostics(torch.cat(feats))
    initial=diagnose(); require_healthy(initial,initial)
    config=dict(method='masked_EMA_representation_ablation',variant=variant,updates=steps,
        batch=batch,lr=1e-5,weight_decay=.05,variance_weight=.1,**settings,
        native256=True,resize=False,mask_range=[.4,.6],validation_images_used=0,
        clear_ec_train_rows=len(train_indices),external_images=0,probe_train_only=True,
        anchor='frozen original CPSAM; visible-position cosine Gram' if anchor is not None else None,
        checkpoint_selection='fixed_terminal',ssl_seed=20260911)
    shared.save_json(destination/'config.json',config)
    shared.save_json(destination/'probe_initial.json',initial)
    start=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    history=[dict(step=0,**initial)]
    for step in range(1,steps+1):
        chosen=np.random.choice(train_indices,size=batch,replace=False)
        assert set(map(int,chosen))<=permitted
        clean=torch.from_numpy(np.stack([views.train(int(i),1)[0] for i in chosen])).cuda()
        mask=torch.from_numpy(mask_batch(batch)).cuda()
        pixel_mask=F.interpolate(mask,size=(256,256),mode='nearest')
        gain=torch.empty(batch,1,1,1,device='cuda').uniform_(.95,1.05)
        offset=torch.empty_like(gain).uniform_(-.02,.02)
        context=(clean*gain+offset).clamp(0,1)*(1-pixel_mask)+.5*pixel_mask
        progress=(step-1)/max(1,steps-1)
        lr=1e-5*min(step/200,1)*(.05+.95*.5*(1+math.cos(math.pi*progress)))
        for group in optimizer.param_groups: group['lr']=lr
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            target=teacher(clean)
            anchor_target=anchor(clean) if anchor is not None else None
        with torch.autocast('cuda',dtype=torch.bfloat16):
            features=encoder(context); prediction=predictor(features)
        errors=F.smooth_l1_loss(normalized_tokens(prediction),normalized_tokens(target),reduction='none').mean(-1)
        prediction_loss=(errors*mask[:,0]).sum()/mask.sum()
        variance,covariance=penalties(features)
        relation=spatial_relation_loss(features,anchor_target,mask) if anchor is not None else features.new_zeros(())
        loss=prediction_loss+.1*variance+settings['covariance_weight']*covariance+settings['anchor_weight']*relation
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite SSL loss')
        loss.backward()
        assert any(p.grad is not None for p in encoder.parameters())
        assert all(p.grad is None for p in teacher.parameters())
        if anchor is not None: assert all(p.grad is None for p in anchor.parameters())
        torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True);optimizer.step()
        ema=.996+(.9999-.996)*progress
        with torch.no_grad():
            for p,q in zip(encoder.parameters(),teacher.parameters()): q.mul_(ema).add_(p,alpha=1-ema)
            for p,q in zip(encoder.buffers(),teacher.buffers()): q.copy_(p)
        if step%25==0 or step==steps:
            elapsed=time.perf_counter()-start
            record=dict(stage='pretraining',variant=variant,step=step,total_steps=steps,
                elapsed_seconds=elapsed,seconds_per_step=elapsed/step,loss=float(loss.detach()),
                prediction_loss=float(prediction_loss.detach()),variance_penalty=float(variance.detach()),
                covariance_penalty=float(covariance.detach()),anchor_penalty=float(relation.detach()),
                peak_gib=torch.cuda.max_memory_allocated()/2**30,updated_at=time.strftime('%FT%T%z'))
            shared.save_json(destination/'progress.json',record);print(record,flush=True)
        if step%500==0 or step==steps:
            stats=diagnose(); require_healthy(stats,initial);history.append(dict(step=step,**stats))
            shared.save_json(destination/'probe_history.json',history)
        if step%2000==0 or step==steps:
            torch.save(dict(teacher_encoder=teacher.state_dict(),config=config,step=step),destination/f'encoder_step{step:04d}.pt')
    torch.cuda.synchronize();final=destination/f'encoder_step{steps:04d}.pt'
    result=dict(updates=steps,elapsed_seconds=time.perf_counter()-start,
        final_checkpoint=str(final),checkpoint_sha256=shared.digest(final),
        initial_probe=initial,final_probe=history[-1],validation_images_used=0)
    shared.save_json(destination/'result.json',result);return result
