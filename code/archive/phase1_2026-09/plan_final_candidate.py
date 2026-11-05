"""Freeze training duration and calibration from completed OOF, before final fit."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from audit_submission_ensembles import sources,metrics,crossfit_scale
from final_candidate_core import save_json

ROOT=Path('experiments/final_candidate_20260904')


def scale_fit(p,y):
    out=[]
    for k in range(3):
        keep=(y[:,k]>0)&np.isfinite(y[:,k]); pp,yy=p[keep,k],y[keep,k]
        ratio=yy/pp; w=pp/yy; ix=np.argsort(ratio)
        out.append(float(ratio[ix[np.searchsorted(np.cumsum(w[ix]),w.sum()/2)]]))
    return np.array(out)


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    src=sources(); groups={42:'fixed_base',77:'base_seed77',2026:'base_seed2026'}
    selection={}; details={}; paths={}
    for seed,group in groups.items():
        detail=[]; paths[str(seed)]=[str(p) for p in src[group]]
        for p in src[group]:
            result=json.loads((p/'result.json').read_text())
            with np.load(p/'split_indices.npz') as d: n=len(d['train'])
            detail.append(dict(train_rows=n,bs=48,best_epoch=result['best_epoch'],
                selected_updates=result['best_epoch']*(n//48)))
        selection[str(seed)]=int(np.median([d['selected_updates'] for d in detail])); details[str(seed)]=detail
    a=Path('experiments/submission_20260904/ensemble_predictions_08_groups.npz')
    with np.load(a,allow_pickle=True) as d:
        p=d['base_three_seeds']; y=d['target']; folds=d['folds']; ids=d['ids'].astype(str)
    scaled,fold_scales=crossfit_scale(p,y,folds); fit=scale_fit(p,y)
    shrink={}
    for alpha in [0.,.5,1.]:
        v=p+alpha*(scaled-p)
        shrink[str(alpha)]=dict(metrics=metrics(v,y).tolist(),overall=float(metrics(v,y).mean()),
            per_fold=[float(metrics(v[folds==f],y[folds==f]).mean()) for f in range(5)])
    # Resample slides, not correlated frames. Conditional uncertainty only.
    with Path('datasets/clear_ec_official/final_train_ids.csv').open(newline='') as f:
        slide={r['ID'].removesuffix('.mha').removesuffix('.bmp'):r['slide_id'] for r in csv.DictReader(f)}
    slide_ids=np.asarray([slide[i] for i in ids]); unique,inverse=np.unique(slide_ids,return_inverse=True)
    valid=(y>0)&np.isfinite(y); denom=np.where(valid,y,1)
    delta=np.where(valid,100*(np.abs(scaled-y)-np.abs(p-y))/denom,0)
    sums=np.zeros((len(unique),3)); counts=np.zeros_like(sums)
    np.add.at(sums,inverse,delta); np.add.at(counts,inverse,valid)
    rng=np.random.default_rng(4092026); boot=[]
    for _ in range(1000):
        ix=rng.integers(len(unique),size=len(unique)); boot.append(sums[ix].sum(0)/counts[ix].sum(0))
    boot=np.array(boot)
    report=dict(n=len(y),global_oof_fit_scale=fit.tolist(),fold_fit_scales=fold_scales,
        shrink_sensitivity=shrink,calibration_delta_by_metric=metrics(scaled,y).tolist(),
        paired_calibration_delta=(metrics(scaled,y)-metrics(p,y)).tolist(),
        conditional_slide_bootstrap_ci=np.quantile(boot,[.025,.975],axis=0).tolist(),
        conditional_overall_delta_ci=np.quantile(boot.mean(1),[.025,.975]).tolist(),
        scope='OOF calibration reused historically; not fully nested; bootstrap conditional on fitted models. Global scale is a candidate setting, not an independent validation score.')
    save_json(ROOT/'calibration_audit.json',report)
    plan=dict(seeds=[42,77,2026],epochs=25,bs=48,selected_steps=selection,selection_details=details,
        fold_paths=paths,calibration=fit.tolist(),weights=[1/3]*3,
        no_final_train_accuracy_selection=True,source_prediction_sha256=hashlib.sha256(a.read_bytes()).hexdigest(),
        jobs='Three all-data fits plus five full-fold inference audits; no challenge submission or pod lifecycle changes.')
    dest=ROOT/'plan.json'
    if dest.exists(): assert json.loads(dest.read_text())==plan
    else: save_json(dest,plan)
    print(json.dumps(dict(selected_steps=selection,scale=fit.tolist(),sensitivity=shrink,
        conditional_overall_delta_ci=report['conditional_overall_delta_ci']),indent=2))


if __name__=='__main__': main()
