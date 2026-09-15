"""Held-out parity/FP16 audit; each process owns one fold and one GPU."""
import argparse
import fcntl
import json
import time
from pathlib import Path
import numpy as np
import torch
from final_candidate_core import build_model,load_member,predict_one,save_json
from audit_submission_ensembles import metrics

ROOT=Path('experiments/final_candidate_20260904')


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--fold',type=int,required=True); args=ap.parse_args()
    assert args.fold in range(5)
    run=ROOT/'inference_audit'/f'fold{args.fold}'; run.mkdir(parents=True,exist_ok=True)
    with (run/'audit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (run/'result.json').exists(): print('already complete'); return
        plan=json.loads((ROOT/'plan.json').read_text())
        torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
        torch.set_float32_matmul_precision('high'); torch.set_num_threads(4)
        meta=np.load('datasets/clear_ec_official/cache_native_meta_512.npz',allow_pickle=True)
        imgs=np.memmap('datasets/clear_ec_official/img_native_512.dat',dtype='uint8',mode='r',shape=(8954,512,512))
        start=time.time(); net=build_model('cpsam_models/cpsam_v2'); init_seconds=time.time()-start
        predictions={}; comparisons={}; timing={}; reference=None
        for seed in plan['seeds']:
            path=Path(plan['fold_paths'][str(seed)][args.fold])
            with np.load(path/'best_oof.npz',allow_pickle=True) as d:
                ix=d['indices']; yy=d['target']; expected=d['pred']; ids=d['ids'].astype(str)
            if reference is None: reference=(ix.copy(),yy.copy(),ids.copy())
            else:
                assert np.array_equal(ix,reference[0]) and np.array_equal(yy,reference[1])
            assert np.array_equal(meta['ids'][ix].astype(str),ids)
            t=time.time(); load_member(net,path/'best.pt'); load_seconds=time.time()-t
            for dtype_name,dtype in [('bf16',torch.bfloat16),('fp16',torch.float16)]:
                predict_one(net,imgs[ix[0]],dtype=dtype); torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(); values=[]; durations=[]
                for k,i in enumerate(ix):
                    t=time.perf_counter(); value=predict_one(net,imgs[i],dtype=dtype)
                    torch.cuda.synchronize(); durations.append(time.perf_counter()-t); values.append(value)
                    if (k+1)%100==0:
                        print(f'seed={seed} {dtype_name} images={k+1}/{len(ix)} seconds={sum(durations):.1f}',flush=True)
                        save_json(run/'progress.json',dict(seed=seed,dtype=dtype_name,images=k+1,total=len(ix),updated_unix=time.time()))
                values=np.array(values); key=f'seed{seed}_{dtype_name}'; predictions[key]=values
                rel=np.abs(values-expected)/expected
                comparisons[key]=dict(mean_relative_difference=rel.mean(0).tolist(),
                    max_relative_difference=rel.max(0).tolist(),metrics=metrics(values,yy).tolist(),
                    overall_score_delta=float(metrics(values,yy).mean()-metrics(expected,yy).mean()))
                timing[key]=dict(mean_seconds=float(np.mean(durations)),p95_seconds=float(np.quantile(durations,.95)),
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,checkpoint_load_seconds=load_seconds)
                assert np.isfinite(values).all() and (values>0).all()
                # Fail closed if small-batch or FP16 execution changes predictions substantially.
                assert rel.mean()<.005 and rel.max()<.05, f'Inference parity outside tolerance: {key}'
        for dt in ['bf16','fp16']:
            predictions['ensemble_'+dt]=np.mean([predictions[f'seed{s}_{dt}'] for s in plan['seeds']],axis=0)
        np.savez_compressed(run/'predictions.npz',indices=reference[0],target=reference[1],ids=reference[2],**predictions)
        save_json(run/'result.json',dict(fold=args.fold,n=len(reference[0]),comparisons=comparisons,timing=timing,
            model_init_seconds=init_seconds,elapsed_seconds=time.time()-start,completed_at=time.strftime('%FT%T%z'),
            device=torch.cuda.get_device_name(),view_batch=8,views=40,
            limits='H200 timings are not T4 certification. Resident single-member timings exclude startup and checkpoint loading.'))


if __name__=='__main__': main()
