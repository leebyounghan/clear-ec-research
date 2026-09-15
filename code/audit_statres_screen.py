"""Paired fixed screen; never tune blend weights, calibrate, or promote candidates."""
import json
import hashlib
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path("experiments/statres_screen_20260908")
OLD = Path("experiments/region_search_20260905/source/code")
sys.path.insert(0,str(OLD))
import audit_region_search as historical
from audit_submission_ensembles import metrics


def main():
    output = ROOT / "screen_audit.json"
    if output.exists(): raise FileExistsError(output)
    complete = json.loads((ROOT / "complete.json").read_text())
    assert not complete["failed"] and sorted(complete["done"]) == [0,1]
    historical.run_path = lambda recipe,fold: Path("experiments/region_seedcheck_20260906/runs") / f"full_quality_f{fold}_seed77_c256k2_mean_multitask_slidesmooth"
    _, old = historical.compare("full_quality",(0,1))
    prior_audit = json.loads(Path("experiments/specialize_screen_20260908/screen_audit.json").read_text())
    hashes = {}
    def record(path,expected=None):
        with Path(path).open("rb") as stream: h = hashlib.file_digest(stream,"sha256").hexdigest()
        if expected is not None: assert h == expected, str(path)
        hashes[str(path)] = h
    # Verify historical baseline provenance has not changed since the last audit.
    for path,h in prior_audit["input_sha256"].items(): record(path,h)
    predictions = {}
    for mode in ("mean","std"):
        pieces = []
        for f in (0,1):
            folder = ROOT / "runs" / f"fold{f}"
            with np.load(folder / mode / "oof.npz") as d:
                mask = old["folds"] == f
                for key in ("ids","indices","target"):
                    np.testing.assert_array_equal(d[key],old[key][mask])
                p = d["pred"].astype(float)
                assert np.isfinite(p).all() and (p>0).all()
                pieces.append(p)
            with np.load(folder / "split_indices.npz") as d, np.load(historical.run_path("full_quality",f) / "split_indices.npz") as r:
                for key in ("train","validation"): np.testing.assert_array_equal(d[key],r[key])
            result = json.loads((folder / mode / "result.json").read_text())
            record(folder / mode / "selected.pt",result["checkpoint_sha256"])
            for rel in (f"{mode}/oof.npz",f"{mode}/result.json","config.json","split_indices.npz"):
                record(folder / rel)
        predictions[mode] = np.concatenate(pieces)
    target, folds, base = old["target"],old["folds"],old["add6"]
    assert len(target)==3582 and len(np.unique(old["ids"]))==3582
    np.testing.assert_allclose(metrics(base,target),prior_audit["current6_metrics"],rtol=0,atol=1e-10)
    scores = {mode:metrics(p,target) for mode,p in predictions.items()}
    blends = {}
    for mode,p in predictions.items():
        blends[mode] = base.copy()
        blends[mode][:,1:] = .5*base[:,1:]+.5*p[:,1:]
        np.testing.assert_array_equal(blends[mode][:,0],base[:,0])
    per_fold=[]
    for f in (0,1):
        m=folds==f
        per_fold.append(dict(fold=f,
            target_cv_hex_mean_delta=float((metrics(predictions['std'][m],target[m])-metrics(predictions['mean'][m],target[m]))[1:].mean()),
            blend_delta_vs_current6=float((metrics(blends['std'][m],target[m])-metrics(base[m],target[m])).mean()),
            blend_delta_vs_control=float((metrics(blends['std'][m],target[m])-metrics(blends['mean'][m],target[m])).mean())))
    delta_target=float((scores['std']-scores['mean'])[1:].mean())
    delta_blend=float((metrics(blends['std'],target)-metrics(base,target)).mean())
    passed=bool(delta_target<=-.03 and delta_blend<=-.01 and all(
        r['target_cv_hex_mean_delta']<0 and r['blend_delta_vs_current6']<0 and r['blend_delta_vs_control']<0 for r in per_fold))
    report=dict(created_at=time.strftime('%FT%T%z'),n=len(target),deployment_approved=False,
        current6_metrics=metrics(base,target).tolist(),current6_overall=float(metrics(base,target).mean()),
        standalone_metrics={k:v.tolist() for k,v in scores.items()},
        std_minus_mean_per_metric=(scores['std']-scores['mean']).tolist(),
        target_cv_hex_mean_delta=delta_target,blend_delta_vs_current6=delta_blend,
        fixed_blend_metrics={k:metrics(v,target).tolist() for k,v in blends.items()},
        fixed_blend_overall={k:float(metrics(v,target).mean()) for k,v in blends.items()},
        per_fold=per_fold,passes_screen=passed,eligible_for_confirmation=['std'] if passed else [],
        caveat='Historically reused OOF; fixed terminal. Feature std is not cell area std. No independent-test or leaderboard claim.')
    manifest=json.loads((ROOT/'manifest.json').read_text())
    for path,h in manifest['source_sha256'].items(): record(path,h)
    for path in (ROOT/'manifest.json',ROOT/'complete.json',Path(__file__)): record(path)
    report['input_sha256']=hashes
    with output.open('x') as f: json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in report.items() if k!='input_sha256'},indent=2))


if __name__=='__main__': main()
