"""Regression distillation contract. Teacher caches are predictions, never GT.

This does not launch a trainer. The bounded continuation plan specifies the
fold-clean cache and matched downstream runner that must be verified first.
"""
import math
import torch
from torch.nn import functional as F


def encoded_teacher(member_predictions, lm, ls):
    """M x B x 3 decoded predictions -> encoded arithmetic teacher mean.

    Decode each member BEFORE averaging, matching the ensemble definition.
    There is no deployment calibration or trainable ensemble weighting here.
    """
    p=member_predictions.detach().float()
    if p.ndim!=3 or p.shape[-1]!=3 or p.shape[0]<2:
        raise ValueError('Expected multiple Bx3 teachers')
    if not torch.isfinite(p).all() or not (p>0).all():
        raise ValueError('Teacher predictions must be finite and positive')
    if lm.shape!=(3,) or ls.shape!=(3,) or not torch.isfinite(lm).all() or not torch.isfinite(ls).all() or not (ls>0).all():
        raise ValueError('Invalid log normalization')
    return (p.mean(0).log()-lm)/ls


def distillation_loss(student, raw_target, teacher_target, log_scale,
                      supervised_loss, weight=0.2):
    if not math.isfinite(weight) or weight<0:
        raise ValueError('Invalid distillation weight')
    if student.shape!=raw_target.shape or student.ndim!=2 or student.shape[-1]!=3:
        raise ValueError('Expected matching Bx3 student and raw target')
    # The supervised term is never downweighted or replaced by pseudo-labels.
    supervised=supervised_loss(student,raw_target,log_scale,'mixed_mape')
    if weight==0:
        return supervised
    if teacher_target.shape!=student.shape or not torch.isfinite(teacher_target).all():
        raise ValueError('Invalid encoded teacher targets')
    return supervised+weight*F.smooth_l1_loss(student.float(),teacher_target.detach().float())


def tests(supervised_loss):
    scale=torch.tensor([.2,.18,.27]);lm=torch.tensor([7.9,-.98,-.62])
    pred=torch.tensor([[[2000.,.3,.6],[2100.,.4,.7]],[[2200.,.4,.5],[1900.,.3,.6]]],requires_grad=True)
    encoded=encoded_teacher(pred,lm,scale)
    torch.testing.assert_close(torch.exp(encoded*scale+lm),pred.detach().mean(0))
    assert not encoded.requires_grad
    raw=torch.zeros((2,3)); student=torch.ones((2,3),requires_grad=True)
    teacher=torch.full((2,3),.5,requires_grad=True)
    base=supervised_loss(student,raw,scale,'mixed_mape')
    zero=distillation_loss(student,raw,teacher,scale,supervised_loss,0.)
    torch.testing.assert_close(base,zero,rtol=0,atol=0)
    a=torch.autograd.grad(base,student,retain_graph=True)[0]
    b=torch.autograd.grad(zero,student,retain_graph=True)[0]
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    full=distillation_loss(student,raw,teacher,scale,supervised_loss)
    full.backward();assert teacher.grad is None and torch.isfinite(student.grad).all()
    for bad in (float('nan'),0.,-1.):
        broken=pred.detach().clone();broken[0,0,0]=bad
        try:encoded_teacher(broken,lm,scale)
        except ValueError:pass
        else:raise AssertionError('Invalid prediction accepted')
    return dict(arithmetic_mean_roundtrip=True,teacher_detached=True,
                zero_weight_loss_and_gradient_exact=True,invalid_teacher_rejected=True)


if __name__=='__main__':
    import json
    import sys
    sys.path.insert(0,'experiments/specialize_screen_20260908/source/frozen')
    from method_training_core import regression_loss
    print(json.dumps(tests(regression_loss)))
