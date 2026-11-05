"""Fixed two-fold add7 comparison; no fitted weights, calibration or promotion."""
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path("experiments/crop_dino_screen_20260910")
sys.path.insert(0,"experiments/region_search_20260905/source/code")
import audit_region_search as historical
from audit_submission_ensembles import metrics


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8*1024**2),b""):h.update(b)
    return h.hexdigest()


def main():
    out=ROOT/"screen_audit.json"
    if out.exists():raise FileExistsError(out)
    complete=json.loads((ROOT/"complete.json").read_text())
    assert not complete["failed"] and len(complete["done"])==4
    historical.run_path=lambda recipe,fold:Path("experiments/region_seedcheck_20260906/runs")/f"full_quality_f{fold}_seed77_c256k2_mean_multitask_slidesmooth"
    _,old=historical.compare("full_quality",(0,1))
    prior=json.loads(Path("experiments/specialize_screen_20260908/screen_audit.json").read_text())
    hashes={}
    for path,h in prior["input_sha256"].items():
        assert sha(path)==h,path
        hashes[path]=h
    target,base,folds=old["target"],old["add6"],old["folds"]
    np.testing.assert_allclose(metrics(base,target),prior["current6_metrics"],rtol=0,atol=1e-10)
    predictions={}
    configs={}
    for policy in ("control","guarded"):
        chunks=[]
        for fold in (0,1):
            folder=ROOT/"runs"/f"{policy}_fold{fold}"
            mask=folds==fold
            with np.load(folder/"oof.npz") as d:
                for key in ("indices","ids","target"):
                    np.testing.assert_array_equal(d[key],old[key][mask])
                chunks.append(d["pred"].astype(float))
            config=json.loads((folder/"config.json").read_text())
            assert not config["smoke"] and config["updates"]==35*(config["train_rows"]//48)
            configs[(policy,fold)]=config
            result=json.loads((folder/"result.json").read_text())
            assert sha(folder/"selected.pt")==result["checkpoint_sha256"]
            for name in ("selected.pt","oof.npz","config.json","result.json","split_indices.npz"):
                hashes[str(folder/name)]=sha(folder/name)
        predictions[policy]=np.concatenate(chunks)
        assert np.isfinite(predictions[policy]).all() and (predictions[policy]>0).all()
    for fold in (0,1):
        for key in ("updates","seed","batch_size","k_crops","split_sha256","initial_sha256","encoder_lr","head_lr","objective","select_mode"):
            assert configs[("control",fold)][key]==configs[("guarded",fold)][key],key
        with np.load(ROOT/"runs"/f"control_fold{fold}/split_indices.npz") as a,np.load(ROOT/"runs"/f"guarded_fold{fold}/split_indices.npz") as b:
            for key in ("train","validation"):np.testing.assert_array_equal(a[key],b[key])
    blended={k:(6*base+p)/7 for k,p in predictions.items()}
    base_score=float(metrics(base,target).mean())
    per_fold=[]
    for fold in (0,1):
        m=folds==fold
        per_fold.append(dict(fold=fold,
            guarded_delta_vs_current6=float((metrics(blended["guarded"][m],target[m])-metrics(base[m],target[m])).mean()),
            guarded_delta_vs_control_blend=float((metrics(blended["guarded"][m],target[m])-metrics(blended["control"][m],target[m])).mean())))
    delta=float(metrics(blended["guarded"],target).mean())-base_score
    passed=bool(delta<=-.02 and all(r["guarded_delta_vs_current6"]<0 and r["guarded_delta_vs_control_blend"]<0 for r in per_fold))
    report=dict(created_at=time.strftime("%FT%T%z"),n=len(target),current6=base_score,
        standalone={k:metrics(p,target).tolist() for k,p in predictions.items()},
        fixed_add7={k:metrics(p,target).tolist() for k,p in blended.items()},
        guarded_delta_vs_current6=delta,per_fold=per_fold,passes_screen=passed,
        eligible_for_confirmation=["guarded"] if passed else [],deployment_approved=False,
        rule="fixed add7=(6*current6+new)/7; guarded improves BOTH folds and aggregate>=0.02 and beats matched control blend in BOTH folds",
        caveat="Historically reused OOF, not an untouched test or significance/leaderboard claim.",input_sha256=hashes)
    with out.open("x") as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in report.items() if k!="input_sha256"},indent=2),flush=True)


if __name__=="__main__":main()
