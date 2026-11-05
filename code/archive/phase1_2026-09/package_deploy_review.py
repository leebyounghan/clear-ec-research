"""Stage corrected candidates only after full numerical audits; never submit."""
import csv
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import SimpleITK as sitk
import torch

from audit_submission_ensembles import metrics, crossfit_scale
from final_candidate_core import central512, save_json
from package_deploy_methods import copy_verified, candidate_predictions
from plan_final_candidate import scale_fit

ROOT = Path("experiments/deploy_review_20260905")
SOURCE = Path(__file__).resolve().parent.parent


def audit_summary():
    entries=[]
    for f in range(5):
        root=ROOT/f"inference_audit/fold{f}"
        result=json.loads((root/"result.json").read_text())
        assert result["n"]>1700 and not result["smoke"]
        with np.load(root/"predictions.npz",allow_pickle=True) as new, np.load(
            f"experiments/final_candidate_20260904/inference_audit/fold{f}/predictions.npz",allow_pickle=True) as old:
            for key in ("indices","target","ids"):
                np.testing.assert_array_equal(new[key],old[key])
            entry={key:new[key].copy() for key in ("indices","target","ids")}
            entry["folds"]=np.full(len(new["indices"]),f)
            for dt in ("bf16","fp16"):
                base=old[f"ensemble_{dt}"].astype(float)
                metric=new[f"metric35_{dt}"].astype(float); bag=new[f"predbag2_35_{dt}"].astype(float)
                entry[f"old3_{dt}"]=base
                entry[f"candidate4_{dt}"]=(base*3+metric)/4
                entry[f"candidate5_{dt}"]=(base*3+metric+bag)/5
            entries.append(entry)
    merged={key:np.concatenate([x[key] for x in entries]) for key in entries[0]}
    assert len(np.unique(merged["indices"]))==8954
    report={}
    for name in ("old3","candidate4","candidate5"):
        for dt in ("bf16","fp16"):
            key=f"{name}_{dt}"; pred=merged[key]; target=merged["target"]
            calibrated,_=crossfit_scale(pred,target,merged["folds"])
            report[key]=dict(raw=metrics(pred,target).tolist(),raw_overall=float(metrics(pred,target).mean()),
                exploratory_crossfit=float(metrics(calibrated,target).mean()))
        drift=report[f"{name}_fp16"]["raw_overall"]-report[f"{name}_bf16"]["raw_overall"]
        if abs(drift)>.01:
            raise ValueError(f"Ensemble precision score drift: {name}: {drift}")
    # Paired slide-cluster bootstrap of fixed raw ensembles; conditional on
    # selected historical OOF models, not a nested/generalization guarantee.
    with open("datasets/clear_ec_official/final_train_ids.csv",newline="") as f:
        mapping={r["ID"].removesuffix(".mha").removesuffix(".bmp"):r["slide_id"] for r in csv.DictReader(f)}
    ids=[str(i).removesuffix(".mha").removesuffix(".bmp") for i in merged["ids"]]
    groups=np.array([mapping[i] for i in ids]); _,inverse=np.unique(groups,return_inverse=True)
    target=merged["target"].astype(float); valid=(target>0)&np.isfinite(target)
    denom=np.where(valid,target,1)
    delta=np.where(valid,(np.abs(merged["candidate5_bf16"]-target)-np.abs(merged["old3_bf16"]-target))/denom*100,0)
    ng=int(inverse.max())+1; sums=np.zeros((ng,3)); counts=np.zeros((ng,3))
    np.add.at(sums,inverse,delta); np.add.at(counts,inverse,valid.astype(float))
    rng=np.random.default_rng(20260905); draws=[]
    for _ in range(2000):
        ix=rng.integers(ng,size=ng); draws.append(float((sums[ix].sum(0)/counts[ix].sum(0)).mean()))
    report["conditional_slide_bootstrap"]=dict(draws=2000,slides=ng,raw_delta=float((sums.sum(0)/counts.sum(0)).mean()),
        interval95=np.quantile(draws,[.025,.975]).tolist(),scope="Conditional on fixed, historically selected OOF models. Does not account for method/checkpoint selection.")
    save_json(ROOT/"inference_summary.json",report)
    np.savez_compressed(ROOT/"inference_predictions.npz",**merged)


def main():
    audit_summary()
    run=ROOT/"runs/predbag2_35"
    result=json.loads((run/"result.json").read_text()); config=json.loads((run/"config.json").read_text())
    assert result["selected_steps"]==3874 and config["initialization"]=="single_head_construction_v2"
    stage=ROOT/"staged"; model=stage/"model"; model.mkdir(parents=True,exist_ok=True)
    files={}; old=Path("experiments/deploy_methods_20260905/staged/model")
    for name in ("cpsam_v2","seed42.pt","seed77.pt","seed2026.pt","metric35.pt"):
        files[f"model/{name}"]=copy_verified(old/name,model/name)
    files["model/predbag2_35.pt"]=copy_verified(run/"selected.pt",model/"predbag2_35.pt")
    for src,name in ((SOURCE/"submission_candidate_20260905/inference.py","inference.py"),
                     (SOURCE/"code/final_candidate_core.py","final_candidate_core.py"),
                     (SOURCE/"submission_candidate_20260905/Dockerfile","Dockerfile"),
                     (SOURCE/"submission_candidate_20260905/requirements.txt","requirements.txt")):
        files[name]=copy_verified(src,stage/name)
    target,predictions=candidate_predictions()
    for name in ("candidate4","candidate5"):
        cfg=json.loads((old/(name+".json")).read_text())
        np.testing.assert_allclose(cfg["calibration"],scale_fit(predictions[name],target),rtol=0,atol=1e-12)
        cfg["initialization_note"]="predbag2 is refitted with the single head initialization used by OOF; no all-data validation score."
        save_json(model/(name+".json"),cfg)
    spec=importlib.util.spec_from_file_location("review_staged",stage/"inference.py")
    api=importlib.util.module_from_spec(spec); spec.loader.exec_module(api)
    meta=np.load("datasets/clear_ec_final9000_20260904/meta.npz",allow_pickle=True)
    indices=np.linspace(0,8999,8,dtype=int); records=[]
    for j,index in enumerate(indices):
        raw=Path("datasets/clear_ec_fullframe_20260904/train_mha")/(str(meta["ids"][index])+".mha")
        image=central512(sitk.GetArrayFromImage(sitk.ReadImage(str(raw))))
        for cfg in ("candidate4.json","candidate5.json"):
            pair={}
            types=[("bf16",torch.bfloat16),("fp16",torch.float16)]
            if j%2: types.reverse()
            for label,dt in types:
                torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
                value=api.predict(image,model,cfg,dt); torch.cuda.synchronize()
                record=dict(index=int(index),candidate=cfg,precision=label,values=value.tolist(),
                    seconds=time.perf_counter()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
                records.append(record); pair[label]=value; print(record,flush=True)
            rel=np.abs(pair["bf16"]-pair["fp16"])/pair["bf16"]
            if rel.max()>.05:
                raise ValueError(f"All-data precision drift exceeds 5%: {cfg}: {rel}")
        if j==0:
            inp=ROOT/"interface_input"; image_dir=inp/"images/corneal-specular-microscopy-image"; image_dir.mkdir(parents=True,exist_ok=True)
            shutil.copy2(raw,image_dir/"image.mha")
            save_json(inp/"inputs.json",[{"socket":{"slug":"corneal-specular-microscopy-image"}}])
    interface_results={}
    for cfg in ("candidate4.json","candidate5.json"):
        out=ROOT/"interface_output"/cfg.removesuffix(".json")
        subprocess.run([sys.executable,str(stage/"inference.py"),"--input",str(ROOT/"interface_input"),"--output",str(out),
            "--model",str(model),"--config",cfg,"--fp16"],check=True,timeout=900)
        values=[json.loads((out/(name+".json")).read_text()) for name in ("cell-density","coefficient-of-variation","hexagonality")]
        if not np.isfinite(values).all() or not (np.array(values)>0).all(): raise ValueError("Invalid interface output")
        interface_results[cfg]=values
    for name in ("candidate4.json","candidate5.json"):
        from package_deploy_methods import digest
        p=model/name; files[f"model/{name}"]=dict(sha256=digest(p),bytes=p.stat().st_size)
    save_json(ROOT/"package_manifest.json",dict(files=files,staged=str(stage),multi_image_native_checks=records,
        interface_results=interface_results,completed_at=time.strftime("%FT%T%z"),
        timing_scope="8 fixed images, alternating precision order; warm Python process, includes per-image model construction/loading. H200, not competition GPU.",
        status="Corrected candidate staged and numerically audited. NOT container-built, target-tested, uploaded or submitted."))


if __name__ == "__main__":
    main()
