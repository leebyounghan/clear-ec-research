"""Predeclared replacement comparison; no weight fitting or promotion."""
import json
from pathlib import Path
import sys
import time
import numpy as np
from prepare_ssl_screen import ROOT, sha, save

sys.path.insert(0, 'experiments/region_search_20260905/source/code')
import audit_region_search as historical
from audit_submission_ensembles import metrics


def main():
    manifest = json.loads((ROOT/'manifest.json').read_text())
    for path, digest in manifest['input_sha256'].items():
        assert sha(path) == digest, path
    for path, digest in manifest['source_sha256'].items():
        assert sha(path) == digest, path
    historical.run_path = lambda recipe, fold: Path('experiments/region_seedcheck_20260906/runs')/f'full_quality_f{fold}_seed77_c256k2_mean_multitask_slidesmooth'
    _, old = historical.compare('full_quality', (0, 1))
    y, base, folds = old['target'], old['add6'], old['folds']
    prior = json.loads(Path('experiments/crop_dino_screen_20260910/screen_audit.json').read_text())
    np.testing.assert_allclose(metrics(base,y).mean(), prior['current6'], rtol=0, atol=1e-10)
    predictions, config, compute, hashes = {}, {}, {}, {}
    for recipe in ('control35', 'ssl35', 'supervised55'):
        chunks = []; times = []
        for fold in (0, 1):
            folder = ROOT/'runs'/f'{"ssl35" if recipe=="ssl35" else "supervised55"}_fold{fold}'
            if recipe=='supervised55':folder=folder/'continued55'
            mask = folds == fold
            with np.load(folder/'oof.npz') as d:
                for key in ('indices', 'ids', 'target'):
                    np.testing.assert_array_equal(d[key], old[key][mask])
                p = d['pred'].astype(float)
                assert np.isfinite(p).all() and (p>0).all()
                chunks.append(p)
            cfg = json.loads((folder/'config.json').read_text())
            config[(recipe,fold)] = cfg
            result = json.loads((folder/'result.json').read_text())
            assert not cfg['smoke'] and result['strict_reload_passed']
            assert sha(folder/'selected.pt') == result['checkpoint_sha256']
            clean_result=json.loads((ROOT/'runs'/f'supervised55_fold{fold}'/'result.json').read_text())
            if recipe!='supervised55':
                assert result['training']['first_three_batches_sha256'] == clean_result['training']['first_three_batches_sha256']
            with np.load(folder/'split_indices.npz') as actual,np.load(ROOT/'preflight'/f'fold{fold}.npz') as expected:
                np.testing.assert_array_equal(actual['train'],expected['train'])
                np.testing.assert_array_equal(actual['validation'],expected['validation'])
            ssl = result.get('pretraining')
            if recipe == 'ssl35':
                assert ssl and ssl['updates'] == 6000 and ssl['validation_images_used'] == 0
                assert sha(ssl['final_checkpoint']) == ssl['checkpoint_sha256']
            times.append(dict(fold=fold, pretrain_seconds=ssl['elapsed_seconds'] if ssl else 0,
                              train_seconds=result['training']['elapsed_seconds']+result.get('initial_training',{}).get('elapsed_seconds',0),
                              eval_seconds=result['evaluation']['elapsed_seconds']+result.get('initial_evaluation',{}).get('elapsed_seconds',0)))
            for name in ('config.json','result.json','oof.npz','selected.pt','split_indices.npz'):
                hashes[str(folder/name)] = sha(folder/name)
        predictions[recipe] = np.concatenate(chunks)
        compute[recipe] = times
    for fold in (0,1):
        for recipe in ('ssl35', 'supervised55'):
            for key in ('seed','batch_size','k_crops','split_sha256','initial_sha256','encoder_lr','head_lr','objective','select_mode'):
                assert config[(recipe,fold)][key] == config[('control35',fold)][key], key
            expected = (35 if recipe=='ssl35' else 55)*(config[(recipe,fold)]['train_rows']//48)
            assert config[(recipe,fold)]['updates'] == expected
    full_replacements = {k:(5*old['current5']+p)/6 for k,p in predictions.items()}
    full_scores=dict(current6=metrics(base,y).tolist(),
                     replacements={k:metrics(p,y).tolist() for k,p in full_replacements.items()})
    # Historical current6 members saw duplicate-linked training groups. Exclude
    # affected validation groups from PRIMARY scoring for the historical comparison.
    primary=[]
    for fold in (0,1):
        with np.load(ROOT/'preflight'/f'fold{fold}.npz') as d:
            primary.extend(d['primary_validation'].tolist())
    mask=np.isin(old['indices'],primary)
    y,base,folds=y[mask],base[mask],folds[mask]
    predictions={k:p[mask] for k,p in predictions.items()}
    replacements={k:p[mask] for k,p in full_replacements.items()}
    comparisons = dict(current6=base, control35=replacements['control35'], supervised55=replacements['supervised55'])
    per_fold=[]
    for fold in (0,1):
        mask=folds==fold
        deltas={k:float((metrics(replacements['ssl35'][mask],y[mask])-metrics(p[mask],y[mask])).mean()) for k,p in comparisons.items()}
        per_fold.append(dict(fold=fold, ssl_replacement_delta=deltas))
    overall_delta=float((metrics(replacements['ssl35'],y)-metrics(base,y)).mean())
    passed=bool(overall_delta<=-.02 and all(all(v<0 for v in f['ssl_replacement_delta'].values()) for f in per_fold))
    paired={}
    for j,name in enumerate(('CD','CV','HEX')):
        valid=y[:,j]>0
        before=abs(base[valid,j]-y[valid,j])/y[valid,j]*100
        after=abs(replacements['ssl35'][valid,j]-y[valid,j])/y[valid,j]*100
        change=after-before
        paired[name]=dict(improved_fraction=float((change<0).mean()),worsened_fraction=float((change>0).mean()),
            improvement_contribution_points=float(np.minimum(change,0).mean()),
            damage_contribution_points=float(np.maximum(change,0).mean()))
    report=dict(created_at=time.strftime('%FT%T%z'), n=len(y), current6=metrics(base,y).tolist(),
        standalone={k:metrics(p,y).tolist() for k,p in predictions.items()},
        fixed_replace6={k:metrics(p,y).tolist() for k,p in replacements.items()},
        descriptive_add7={k:metrics((6*base+p)/7,y).tolist() for k,p in predictions.items()},
        ssl_replace6_delta_vs_current6=overall_delta, per_fold=per_fold,
        paired_error_changes=paired, compute=compute, passes_screen=passed,
        full_historical_validation_descriptive_only=full_scores,
        eligible_for_review=['ssl35'] if passed else [], deployment_approved=False,
        caveat='Historically reused OOF; primary validation excludes duplicate-linked slide groups seen by historical baseline training. New training also excludes duplicate-linked validation groups. Continued55 resets optimizer after clean35 and is approximate compute control, not exact FLOP matching. No significance or leaderboard claim.',
        input_sha256=hashes)
    save(ROOT/'screen_audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='input_sha256'}),flush=True)


if __name__=='__main__':
    main()
