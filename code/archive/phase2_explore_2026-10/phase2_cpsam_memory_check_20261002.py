"""Gate query-chunking on numerical parity before full-native resource tests.

No optimization of model weights, data labels, official artifacts or predictions
on a validation set. H200 measurements never certify T4 deployment.
"""
import argparse
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from phase2_foundation_preflight_20261002 import build, save
from cpsam_chunked_attention_20261002 import install_encoder, parity_check

ROOT = Path('experiments/phase2_foundation_optimization_20261002')


def image(size):
    prior = json.loads(Path('experiments/phase2_foundation_20261002/preflight_manifest.json').read_text())
    raw = np.memmap('datasets/clear_ec_fullframe_20260904/img_full_raw.dat', mode='r',
                    dtype=np.uint8, shape=(prior['total_images'], 972, 1296))
    view = np.array(raw[prior['probe_indices'][0]], copy=True).astype(np.float32)/255
    if size:
        h,w = view.shape
        view = view[(h-size)//2:(h-size)//2+size, (w-size)//2:(w-size)//2+size]
    x = torch.from_numpy(np.stack([view]*3)[None]).cuda()
    return F.pad(x,(0,(-x.shape[-1])%8,0,(-x.shape[-2])%8),mode='replicate')


def comparison(reference, candidate, x, dtype, gradients):
    results=[]
    parameter_gradient = None
    input_gradient = None
    for model in (reference,candidate):
        model.zero_grad(set_to_none=True)
        xi=x.detach().clone().requires_grad_(gradients)
        model.train(gradients)
        torch.manual_seed(4242)
        with torch.set_grad_enabled(gradients):
            with torch.autocast('cuda',dtype=dtype,enabled=dtype != torch.float32):
                output=model(xi)
            if gradients:
                output.float().square().mean().backward()
        results.append(output.detach().float())
        if gradients and parameter_gradient is None:
            parameter_gradient={k:p.grad.detach().clone() for k,p in model.named_parameters() if p.grad is not None}
            input_gradient=xi.grad.detach().clone()
        elif gradients:
            error_square, reference_square = [], []
            maximum=0.0
            for k,p in model.named_parameters():
                if p.grad is None:
                    assert k not in parameter_gradient
                    continue
                g=parameter_gradient.pop(k)
                delta=(p.grad-g).float()
                error_square.append(delta.square().sum())
                reference_square.append(g.float().square().sum())
                maximum=max(maximum,delta.abs().max().item())
            assert not parameter_gradient
            relative_l2=(torch.stack(error_square).sum()/torch.stack(reference_square).sum().clamp_min(1e-30)).sqrt().item()
            input_relative=(torch.linalg.vector_norm(xi.grad-input_gradient)/torch.linalg.vector_norm(input_gradient).clamp_min(1e-30)).item()
    delta=results[0]-results[1]
    rel=(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(results[0]).clamp_min(1e-30)).item()
    report=dict(dtype=str(dtype),input=list(x.shape),gradients=gradients,
                output_max_abs=delta.abs().max().item(),output_relative_l2=rel)
    limit = 0.001 if dtype==torch.float32 else 0.03
    if gradients:
        report.update(parameter_gradient_relative_l2=relative_l2,
                      parameter_gradient_max_abs=maximum,input_gradient_relative_l2=input_relative)
    report['gate_relative_l2_limit']=limit
    report['passed']=bool(torch.isfinite(delta).all().item() and rel<=limit and
                          (not gradients or (relative_l2<=limit and input_relative<=limit)))
    reference.zero_grad(set_to_none=True); candidate.zero_grad(set_to_none=True)
    return report


def parity():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    unit=parity_check('cuda')
    torch.manual_seed(42)
    reference,_,details=build('cpsam',True)
    reference=reference.cuda()
    candidate=deepcopy(reference)
    install_encoder(candidate[0].original, query_chunk=256, checkpoint_chunks=True)
    assert reference.state_dict().keys()==candidate.state_dict().keys()
    tests=[]
    for dtype in (torch.float32,torch.bfloat16,torch.float16):
        for size, gradients in ((256,True),(512,False)):
            test=comparison(reference,candidate,image(size),dtype,gradients)
            tests.append(test)
            save(ROOT/'parity_progress.json',dict(tests=tests,unit_tests=unit))
            print(json.dumps(test),flush=True)
    result=dict(unit_tests=unit,tests=tests,all_passed=all(x['passed'] for x in tests),
                foundation=details,source_sha256=hashlib.file_digest(Path(__file__).open('rb'),'sha256').hexdigest(),
                chunk_source_sha256=hashlib.file_digest(Path('experiments/phase2_foundation_optimization_20261002/source/cpsam_chunked_attention_20261002.py').open('rb'),'sha256').hexdigest(),
                limits='Small-unit and 256 gradient / 512 output parity; no accuracy guarantee or T4 certification.')
    save(ROOT/'parity.json',result)
    if not result['all_passed']:
        raise RuntimeError('Numerical parity gate failed; no full-native benchmark permitted')


def benchmark():
    gate=json.loads((ROOT/'parity.json').read_text())
    if not gate['all_passed']:
        raise ValueError('Parity gate missing or failed')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=True
    net,_,details=build('cpsam',True)
    install_encoder(net[0].original,query_chunk=256,checkpoint_chunks=True)
    net=net.cuda().eval()
    x=image(None)
    timings=[]
    for dtype in (torch.bfloat16,torch.float16):
        with torch.inference_mode():
            for step in range(4):
                torch.cuda.synchronize()
                if step==1:
                    torch.cuda.reset_peak_memory_stats()
                t0=time.perf_counter()
                with torch.autocast('cuda',dtype=dtype):
                    out=net(x)
                torch.cuda.synchronize()
                if step>0:
                    timings.append(time.perf_counter()-t0)
                if not torch.isfinite(out).all().item():
                    raise RuntimeError('Non-finite full-native output')
        result=dict(dtype=str(dtype),input=list(x.shape),gpu=torch.cuda.get_device_name(),
                    inference_seconds=float(np.median(timings)),
                    inference_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    inference_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                    query_chunk=256,foundation=details,
                    note='H200 only; full KV and positional terms preserved. No trained weights changed.')
        save(ROOT/('fullnative_'+str(dtype).split('.')[-1]+'.json'),result)
        print(json.dumps(result),flush=True)
        timings=[]
    save(ROOT/'complete.json',dict(status='completed',t4_tested=False,accuracy_training_started=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['parity','benchmark'])
    args=parser.parse_args()
    parity() if args.mode=='parity' else benchmark()
