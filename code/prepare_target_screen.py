"""CPU preflight: immutable controls, clean splits, training-only target audit."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from target_screen_core import summarize_target_shift

ROOT=Path('experiments/target_screen_20260912')
PRIOR=Path('experiments/ssl_screen_20260911')
INITIAL='0f1cc3f7ecdd8a037a57c6c48d9d8921391be4cbce3fa9f13c3e3a2e1253c667'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):
            h.update(block)
    return h.hexdigest()


def save_new(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,indent=2,allow_nan=False)


def main():
    pre=ROOT/'preflight'; pre.mkdir(exist_ok=False)
    prior=json.loads((PRIOR/'manifest.json').read_text())
    completed=json.loads((PRIOR/'queue_finished.json').read_text())
    audit=json.loads((PRIOR/'screen_audit.json').read_text())
    assert completed['audit_completed'] and sha('cpsam_models/cpsam_v2')==INITIAL
    inputs={}
    for records in (prior['input_sha256'],audit['input_sha256']):
        for p,h in records.items():
            assert sha(p)==h,p
            inputs[p]=h
    sources=dict(prior['source_sha256'])
    for p,h in sources.items():
        assert sha(p)==h,p
    data=Path('datasets/clear_ec_official')
    with np.load(data/'cache_native_meta_512.npz',allow_pickle=True) as meta:
        ids,y=meta['ids'].astype(str),meta['Y'].astype(np.float32)
    with (data/'final_train_ids.csv').open() as f:
        mapping={Path(r['ID']).stem:str(r['slide_id']) for r in csv.DictReader(f)}
    slides=np.array([mapping[Path(i).stem] for i in ids])
    frame=Path('datasets/clear_ec_fullframe_20260904')
    frame_report=json.loads((frame/'fullframe_manifest.json').read_text())
    images=np.memmap(frame/'img_full_raw.dat',mode='r',dtype=np.uint8,
                     shape=tuple(frame_report['cache_shape']))
    with np.load(PRIOR/'preflight/pixel_fingerprints.npz') as d:
        fingerprints=d['sha256'].copy()
    assert len(ids)==8954 and len(fingerprints)==len(ids)
    start=time.monotonic()
    for i in range(len(ids)):
        assert hashlib.sha256(images[i].tobytes()).hexdigest()==fingerprints[i], 'Raw image cache changed'
        if (i+1)%2000==0:
            print(json.dumps(dict(stage='pixel_integrity',images=i+1,seconds=time.monotonic()-start)),flush=True)
    parents={s:s for s in slides}
    def find(s):
        while parents[s]!=s:
            parents[s]=parents[parents[s]];s=parents[s]
        return s
    seen={}
    for slide,h in zip(slides,fingerprints):
        if h in seen:parents[find(slide)]=find(seen[h])
        else:seen[h]=slide
    connected=np.array([find(s) for s in slides])
    # This compare verifies identical historical training and validation indices
    # across ALL six ensemble members, not just the regional sixth member.
    sys.path.insert(0,'experiments/region_search_20260905/source/code')
    import audit_region_search as historical
    from audit_submission_ensembles import metrics
    historical.run_path=lambda recipe,fold:Path('experiments/region_seedcheck_20260906/runs')/f'full_quality_f{fold}_seed77_c256k2_mean_multitask_slidesmooth'
    _,reference=historical.compare('full_quality',(0,1))
    records={}; primary=[]
    for fold in (0,1):
        split_path=PRIOR/'preflight'/f'fold{fold}.npz'
        assert sha(split_path)==prior['folds'][str(fold)]['split_sha256']
        with np.load(split_path) as d:
            tr,va,primary_va,old_tr=[d[k].copy() for k in ('train','validation','primary_validation','historical_train')]
        assert not set(connected[tr])&set(connected[va])
        assert not set(connected[old_tr])&set(connected[primary_va])
        assert set(primary_va)<=set(va)
        primary.extend(primary_va.tolist())
        expected=reference['folds']==fold
        np.testing.assert_array_equal(ids[va],reference['ids'][expected])
        np.testing.assert_array_equal(y[va],reference['target'][expected])
        controls={}
        for initialization,folder_name in (('base','supervised55'),('ssl','ssl35')):
            control=PRIOR/'runs'/f'{folder_name}_fold{fold}'
            config=json.loads((control/'config.json').read_text())
            result=json.loads((control/'result.json').read_text())
            assert config['updates']==35*(len(tr)//48) and config['seed']==77
            assert config['initial_sha256']==INITIAL and not config['smoke']
            assert result['strict_reload_passed']
            with np.load(control/'split_indices.npz') as d:
                np.testing.assert_array_equal(tr,d['train'])
                np.testing.assert_array_equal(va,d['validation'])
            with np.load(control/'oof.npz') as d:
                np.testing.assert_array_equal(d['indices'],va)
                np.testing.assert_array_equal(d['target'],y[va])
            controls[initialization]=dict(folder=str(control),
                sampling_sha256=result['training']['first_three_batches_sha256'],
                config_sha256=sha(control/'config.json'),checkpoint_sha256=result['checkpoint_sha256'])
            if initialization=='ssl':
                ssl=result['pretraining']
                assert ssl['updates']==6000 and ssl['validation_images_used']==0
                assert sha(ssl['final_checkpoint'])==ssl['checkpoint_sha256']
                controls[initialization]['encoder_path']=ssl['final_checkpoint']
                controls[initialization]['encoder_sha256']=ssl['checkpoint_sha256']
                inputs[ssl['final_checkpoint']]=ssl['checkpoint_sha256']
        assert controls['base']['sampling_sha256']==controls['ssl']['sampling_sha256']
        summary=summarize_target_shift(y,slides,tr)
        assert summary['changed_rows']>0,'No target change: training would be redundant'
        records[str(fold)]=dict(**summary,validation_rows=len(va),primary_rows=len(primary_va),
            split_path=str(split_path),split_sha256=sha(split_path),controls=controls,
            exact_duplicate_and_slide_overlap=0)
    primary=np.isin(reference['indices'],primary)
    np.testing.assert_allclose(metrics(reference['add6'][primary],reference['target'][primary]),audit['current6'],rtol=0,atol=1e-10)
    np.savez_compressed(pre/'reference.npz',**reference,primary=primary)
    inputs[str(pre/'reference.npz')]=sha(pre/'reference.npz')
    for p in (PRIOR/'manifest.json',PRIOR/'screen_audit.json',PRIOR/'queue_finished.json',
              PRIOR/'preflight/pixel_fingerprints.npz'):
        inputs[str(p)]=sha(p)
    sources.update({str(p):sha(p) for p in (ROOT/'source').glob('*.py')})
    manifest=dict(created_at=time.strftime('%FT%T%z'),initial_sha256=INITIAL,
        source_sha256=sources,input_sha256=inputs,folds=records,gpu_count=4,
        recipes=['base_raw','ssl_raw'],target_policy='raw',epochs=35,seed=77,
        max_queue_seconds=7*3600,max_worker_seconds=6*3600,smoke_timeout_seconds=1200,
        primary_validation_rows=int(primary.sum()),current6_metrics=audit['current6'],
        research_gate='Standalone improves matched smoothed BOTH folds and aggregate>=0.02 points',
        ensemble_gate='Fixed replace6 improves current6 BOTH folds and aggregate>=0.02; beats matched smoothed replace6 BOTH folds',
        multiple_hypotheses=2,statistical_significance_claim=False,
        no_new_ssl_training=True,external_data_used=False,new_weights_downloaded=False,
        deployment_approved=False,automatic_confirmation=False,historical_validation_reuse=True,
        limitations=['slide_id is not established patient identity','near duplicates not exhaustively audited',
                    'same historically reused validation; future confirmation needs fresh fold-specific initializers'])
    save_new(ROOT/'manifest.json',manifest)
    save_new(pre/'target_audit.json',dict(created_at=manifest['created_at'],folds={f:{k:v for k,v in r.items() if k!='controls'} for f,r in records.items()},
                                      exact_unique_images=len(set(fingerprints)),n=len(ids),
                                      comparison_source='training rows only',no_per_image_data_exported=True))
    print(json.dumps(dict(preflight_passed=True,gpu_count=4,folds={f:{k:v for k,v in r.items() if k!='controls'} for f,r in records.items()})),flush=True)


if __name__=='__main__':main()
