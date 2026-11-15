"""Frozen-token linear point probe. No encoder updates or cell-mask supervision."""
import torch
import numpy as np
from green_point_spatial_20261023 import OFFSET, STRIDE, match_points

def build_head(seed=46):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return torch.nn.Linear(768,1)

def loss(head, features, target, support):
    assert features.ndim==3 and features.shape[-1]==768
    assert features.shape[:2]==target.shape==support.shape
    assert support.dtype==torch.bool and support.any()
    assert not features.requires_grad, 'Encoder tokens must be detached'
    logits=head(features.float()).squeeze(-1)
    return torch.nn.functional.binary_cross_entropy_with_logits(logits[support],target[support].float())

def peaks(probability, support, threshold=0.5, radius=16.0):
    # Deterministic score order; greedy physical-distance NMS also handles plateaus.
    y,x=np.where(support & (probability>=threshold))
    order=np.lexsort((x,y,-probability[y,x]))
    found=[]
    for i in order:
        p=np.array([OFFSET+STRIDE*x[i],OFFSET+STRIDE*y[i]])
        if all(np.linalg.norm(p-q)>radius for q in found):found.append(p)
    return np.asarray(found,dtype=float).reshape(-1,2)

def cpu_tests():
    torch.set_num_threads(1)
    h=build_head();h2=build_head()
    assert all(torch.equal(a,b) for a,b in zip(h.parameters(),h2.parameters()))
    gen=torch.Generator().manual_seed(123)
    x=torch.randn(4,5,768,generator=gen);t=torch.rand(4,5,generator=gen)
    s=torch.zeros(4,5,dtype=torch.bool);s[1:3,1:4]=True
    before=x.clone();l=loss(h,x,t,s)
    # Values outside annotated support must never contribute to loss or gradients.
    x2=x.clone();x2[~s]=999;t2=t.clone();t2[~s]=1
    assert torch.equal(l,loss(h,x2,t2,s))
    old=h.weight.detach().clone();opt=torch.optim.AdamW(h.parameters(),lr=.001,weight_decay=.0001)
    opt.zero_grad();l.backward();assert all(torch.isfinite(p.grad).all() for p in h.parameters());opt.step()
    assert not torch.equal(old,h.weight) and torch.equal(before,x)
    a=np.zeros((4,5));a[1,1]=.9;a[1,2]=.8;a[3,4]=.7
    pts=peaks(a,np.ones_like(a,dtype=bool))
    assert len(pts)==2
    assert match_points(pts,[[11.5,11.5],[35.5,27.5]])['f1']==1
    return dict(status='passed',parameters=sum(p.numel() for p in h.parameters()),
        checks=['deterministic_seed','ROI_masked_loss_invariance','finite_gradients',
        'head_update','features_unchanged','physical_NMS','one_to_one_detection'],
        encoder_training=False,heldout_used_for_training=False)
if __name__=='__main__':
    import json
    print(json.dumps(cpu_tests()))
