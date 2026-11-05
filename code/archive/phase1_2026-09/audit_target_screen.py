"""Fixed 2x2 comparison: encoder initialization x target averaging policy."""
import json
from pathlib import Path
import sys
import time
import numpy as np
from prepare_target_screen import ROOT, sha, save_new
from target_screen_core import decision
sys.path.insert(0,'experiments/region_search_20260905/source/code')
from audit_submission_ensembles import metrics


def main():
    manifest=json.loads((ROOT/'manifest.json').read_text())
    complete=json.loads((ROOT/'complete.json').read_text())
    assert not complete['failed'] and len(complete['done'])==4
    for records in (manifest['source_sha256'],manifest['input_sha256']):
        for p,h in records.items():assert sha(p)==h,p
    with np.load(ROOT/'preflight/reference.npz') as d:
        ref={k:d[k].copy() for k in d.files}
    primary=ref['primary'];target=ref['target'][primary];folds=ref['folds'][primary]
    base=ref['add6'][primary];old5=ref['current5'][primary]
    np.testing.assert_allclose(metrics(base,target),manifest['current6_metrics'],rtol=0,atol=1e-10)
    predictions={};hashes={};compute={};full_scores={}; configs={}
    for initialization in ('base','ssl'):
        for policy in ('raw','slidesmooth'):
            name=f'{initialization}_{policy}';chunks=[];times=[]
            for fold in (0,1):
                record=manifest['folds'][str(fold)];control=record['controls'][initialization]
                folder=ROOT/'runs'/f'{name}_fold{fold}' if policy=='raw' else Path(control['folder'])
                config=json.loads((folder/'config.json').read_text())
                result=json.loads((folder/'result.json').read_text())
                assert not config['smoke'] and result['strict_reload_passed']
                assert config['updates']==35*(record['train_rows']//48)
                assert config['split_sha256']==record['split_sha256']
                assert result['training']['first_three_batches_sha256']==control['sampling_sha256']
                assert sha(folder/'selected.pt')==result['checkpoint_sha256']
                configs[(initialization,policy,fold)]=config
                with np.load(record['split_path']) as expected,np.load(folder/'split_indices.npz') as actual:
                    for key in ('train','validation'):np.testing.assert_array_equal(actual[key],expected[key])
                with np.load(folder/'oof.npz') as d:
                    mask=ref['folds']==fold
                    for key in ('indices','ids','target'):np.testing.assert_array_equal(d[key],ref[key][mask])
                    p=d['pred'].astype(float)
                    assert np.isfinite(p).all() and (p>0).all()
                    chunks.append(p)
                for filename in ('config.json','result.json','selected.pt','split_indices.npz','oof.npz'):
                    hashes[str(folder/filename)]=sha(folder/filename)
                times.append(dict(fold=fold,train_seconds=result['training']['elapsed_seconds'],
                                  eval_seconds=result['evaluation']['elapsed_seconds'],
                                  new_compute_this_queue=(policy=='raw')))
            all_predictions=np.concatenate(chunks)
            full_scores[name]=metrics(all_predictions,ref['target']).tolist()
            predictions[name]=all_predictions[primary];compute[name]=times
    for initialization in ('base','ssl'):
        for fold in (0,1):
            a=configs[(initialization,'raw',fold)];b=configs[(initialization,'slidesmooth',fold)]
            for key in ('updates','seed','k_crops','batch_size','encoder_lr','head_lr','objective','region_policy',
                        'select_mode','initial_sha256','ssl_checkpoint_sha256','lm','ls'):
                assert a[key]==b[key],key
    blended={k:(5*old5+p)/6 for k,p in predictions.items()}
    results={}
    for initialization in ('base','ssl'):
        raw=f'{initialization}_raw';smooth=f'{initialization}_slidesmooth'
        def deltas(mask):
            return dict(standalone_vs_matched=float((metrics(predictions[raw][mask],target[mask])-metrics(predictions[smooth][mask],target[mask])).mean()),
                replace6_vs_matched=float((metrics(blended[raw][mask],target[mask])-metrics(blended[smooth][mask],target[mask])).mean()),
                replace6_vs_current6=float((metrics(blended[raw][mask],target[mask])-metrics(base[mask],target[mask])).mean()))
        overall=deltas(np.ones(len(target),bool))
        per_fold=[dict(fold=f,**deltas(folds==f)) for f in (0,1)]
        results[initialization]=dict(aggregate=overall,per_fold=per_fold,
                                    **decision(overall,per_fold))
    interaction={}
    for space,values in (('standalone',predictions),('replace6',blended)):
        interaction[space]=dict(ssl_gain_with_raw=float((metrics(values['ssl_raw'],target)-metrics(values['base_raw'],target)).mean()),
            ssl_gain_with_smoothing=float((metrics(values['ssl_slidesmooth'],target)-metrics(values['base_slidesmooth'],target)).mean()))
        interaction[space]['difference_of_deltas']=interaction[space]['ssl_gain_with_raw']-interaction[space]['ssl_gain_with_smoothing']
    report=dict(completed_at=time.strftime('%FT%T%z'),n=len(target),current6=metrics(base,target).tolist(),
        standalone={k:metrics(p,target).tolist() for k,p in predictions.items()},
        fixed_replace6={k:metrics(p,target).tolist() for k,p in blended.items()},
        comparisons=results,ssl_target_policy_interaction=interaction,compute=compute,
        full_historical_validation_descriptive_only=full_scores,
        eligible_for_research=[k for k,v in results.items() if v['research_signal']],
        eligible_for_ensemble_confirmation=[k for k,v in results.items() if v['passes_ensemble_screen']],
        input_sha256=hashes,deployment_approved=False,automatic_confirmation=False,
        caveat='Two planned hypotheses; reused slide-group OOF and fixed duplicate-clean primary subset, not independent test. A smoothing benefit does not prove labels correct; a raw benefit does not prove label noise absent. No tuned weights, calibration or leaderboard forecast.')
    save_new(ROOT/'screen_audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='input_sha256'}),flush=True)


if __name__=='__main__':main()
