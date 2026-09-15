"""Same-scope postflight: stage immutable model files and test the real entrypoint."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
from audit_submission_ensembles import metrics,crossfit_scale
from final_candidate_core import save_json

ROOT=Path('experiments/final_candidate_20260904')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()


def main():
    plan=json.loads((ROOT/'plan.json').read_text())
    indices=[]; ys=[]; bf=[]; fp=[]; folds=[]; timing=[]
    for fold in range(5):
        r=ROOT/'inference_audit'/f'fold{fold}'
        report=json.loads((r/'result.json').read_text()); timing.append(report['timing'])
        with np.load(r/'predictions.npz',allow_pickle=True) as d:
            indices.extend(d['indices']); ys.append(d['target']); bf.append(d['ensemble_bf16']); fp.append(d['ensemble_fp16'])
            folds.extend([fold]*len(d['indices']))
    assert len(indices)==len(set(indices))==8954
    y=np.concatenate(ys); pp=np.concatenate(fp); bb=np.concatenate(bf); folds=np.asarray(folds)
    report={}
    for name,p in [('bf16',bb),('fp16',pp)]:
        corrected,_=crossfit_scale(p,y,folds)
        report[name]=dict(raw=metrics(p,y).tolist(),raw_overall=float(metrics(p,y).mean()),
            crossfit=metrics(corrected,y).tolist(),crossfit_overall=float(metrics(corrected,y).mean()))
    report['timing']=timing
    report['scope']='Held-out inference parity only. H200 timing is not target-device certification. Final all-data model has no held-out score.'
    save_json(ROOT/'inference_summary.json',report)
    staging=ROOT/'staged'; model=staging/'model'; model.mkdir(parents=True,exist_ok=True)
    manifest={}
    for src,name in [(Path('cpsam_models/cpsam_v2'),'cpsam_v2')]+[
            (ROOT/'runs'/f'seed{s}'/'selected.pt',f'seed{s}.pt') for s in plan['seeds']]:
        expected=digest(src); dest=model/name
        if dest.exists(): assert digest(dest)==expected
        else: shutil.copy2(src,dest)
        manifest[name]=dict(sha256=expected,bytes=dest.stat().st_size)
    save_json(model/'candidate.json',dict(seeds=plan['seeds'],members=[f'seed{s}.pt' for s in plan['seeds']],
        calibration=plan['calibration'],view_batch=8,selected_steps=plan['selected_steps'],
        note='All-data candidate; no leaderboard score or all-data validation claim.'))
    for src,name in [(Path('submission_candidate_20260904/inference.py'),'inference.py'),
                     (Path('code/final_candidate_core.py'),'final_candidate_core.py')]:
        dest=staging/name
        if dest.exists(): assert digest(dest)==digest(src)
        else: shutil.copy2(src,dest)
    # Exercise actual file/JSON interface on an original image, FP16 and BF16 paths.
    data=np.load('datasets/clear_ec_final9000_20260904/meta.npz')
    source=Path('datasets/clear_ec_fullframe_20260904/train_mha')/(str(data['ids'][0])+'.mha')
    inp=ROOT/'interface_smoke/input'; folder=inp/'images/corneal-specular-microscopy-image'; folder.mkdir(parents=True,exist_ok=True)
    image=folder/'image.mha'
    if not image.exists(): shutil.copy2(source,image)
    save_json(inp/'inputs.json',[dict(socket=dict(slug='corneal-specular-microscopy-image'))])
    smokes={}
    for name,extra in [('bf16',[]),('fp16',['--fp16'])]:
        out=ROOT/'interface_smoke'/name
        if out.exists(): raise FileExistsError('Inspect prior interface smoke before retry')
        start=time.time()
        subprocess.run([sys.executable,str(staging/'inference.py'),'--input',str(inp),'--output',str(out),
            '--model',str(model)]+extra,check=True,timeout=600,env=dict(os.environ,OMP_NUM_THREADS='4'))
        values=[json.loads((out/(s+'.json')).read_text()) for s in ['cell-density','coefficient-of-variation','hexagonality']]
        assert np.isfinite(values).all() and (np.asarray(values)>0).all()
        smokes[name]=dict(values=values,cold_process_seconds=time.time()-start)
    save_json(ROOT/'package_manifest.json',dict(files=manifest,interface_smoke=smokes,
        staged=str(staging),completed_at=time.strftime('%FT%T%z'),
        status='Staged and native entrypoint tested. NOT Docker built, target-device tested, uploaded or submitted.'))
    print(json.dumps(dict(ready=str(staging),scores=report,smoke=smokes),indent=2),flush=True)


if __name__=='__main__': main()
