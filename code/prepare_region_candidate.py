"""Separate experimental stage and two native integration jobs; never submit."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path("experiments/region_candidate_20260907")
FROZEN = Path("experiments/region_search_20260905/source/code")
sys.path.append(str(FROZEN))
import numpy as np
from final_candidate_core import save_json
from plan_final_candidate import scale_fit
from audit_submission_ensembles import metrics, crossfit_scale


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8*1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def worker(policy):
    import torch
    import SimpleITK as sitk
    stage = ROOT / "experimental_stage"
    sys.path.insert(0, str(stage))
    spec = importlib.util.spec_from_file_location("region_candidate", stage / "inference.py")
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    import base_inference
    from final_candidate_core import central512
    out = ROOT / "integration" / policy
    out.mkdir(parents=True, exist_ok=False)
    cfg_name = f"candidate6_{policy}.json"
    cfg = json.loads((stage / "model" / cfg_name).read_text())
    api.validate(cfg, cfg_name)
    for change in (dict(region_policy="invalid"), dict(members=cfg["members"][:-1]), dict(calibration=[float("nan"),1,1]), dict(view_batch=40)):
        try:
            api.validate(dict(cfg, **change), cfg_name)
        except ValueError:
            pass
        else:
            raise AssertionError("Malformed candidate config accepted")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    meta = np.load("datasets/clear_ec_final9000_20260904/meta.npz")
    records = []
    first = None
    for j, index in enumerate(np.linspace(0,8999,8,dtype=int)):
        raw = Path("datasets/clear_ec_fullframe_20260904/train_mha") / (str(meta["ids"][index])+".mha")
        image = sitk.GetArrayFromImage(sitk.ReadImage(str(raw)))
        pair = {}
        precision = [("bf16",torch.bfloat16),("fp16",torch.float16)]
        if j % 2:
            precision.reverse()
        for label, dtype in precision:
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            result, members = api.predict(image, stage/"model", cfg_name, dtype, return_members=True)
            torch.cuda.synchronize()
            np.testing.assert_allclose(result, members.mean(0)*cfg["calibration"], rtol=0, atol=0)
            records.append(dict(index=int(index), precision=label, values=result.tolist(), seconds=time.perf_counter()-start,
                                peak_gib=torch.cuda.max_memory_allocated()/1024**3))
            pair[label] = result
            if j == 0:
                old = Path("experiments/deploy_review_20260905/staged/model")
                previous = base_inference.predict(central512(image), old, "candidate5.json", dtype)
                old_scale = json.loads((old/"candidate5.json").read_text())["calibration"]
                np.testing.assert_allclose(members[:5].mean(0), previous/old_scale, rtol=1e-6, atol=1e-6)
        assert (np.abs(pair["bf16"]-pair["fp16"])/pair["bf16"]).max() <= .05
        if j == 0:
            first = pair
            inp = out / "input"
            images = inp / "images/corneal-specular-microscopy-image"
            images.mkdir(parents=True)
            shutil.copy2(raw, images / "image.mha")
            save_json(inp / "inputs.json", [{"socket":{"slug":"corneal-specular-microscopy-image"}}])
        save_json(out / "progress.json", dict(images=j+1,total=8,updated_at=time.strftime("%FT%T%z")))
    entries = {}
    for precision, flags in (("bf16",[]),("fp16",["--fp16"])):
        destination = out / f"output_{precision}"
        command = [sys.executable,str(stage/"inference.py"),"--input",str(out/"input"),"--output",str(destination),"--model",str(stage/"model"),"--config",cfg_name,*flags]
        start = time.time()
        result = subprocess.run(command,check=True,capture_output=True,text=True,timeout=300)
        values = [json.loads((destination/f"{name}.json").read_text()) for name in ("cell-density","coefficient-of-variation","hexagonality")]
        assert len(list(destination.glob("*.json"))) == 3
        np.testing.assert_allclose(values, first[precision], rtol=1e-5, atol=1e-6)
        entries[precision] = dict(values=values,cold_process_seconds=time.time()-start,stdout=result.stdout)
    save_json(out / "result.json", dict(policy=policy,records=records,entrypoints=entries,completed_at=time.strftime("%FT%T%z"),
        deployment_approved=False,scope="Eight fixed native images; warm six-member timings include construction/loading, not container or target device certification"))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--worker",choices=("anchor_quality","full_quality"))
    args=parser.parse_args()
    if args.worker:
        return worker(args.worker)
    if (ROOT/"stage_manifest.json").exists():
        raise FileExistsError("Already prepared; do not rerun")
    audit=Path("experiments/region_inference_20260907")
    complete=json.loads((audit/"complete.json").read_text())
    assert not complete["failed"] and len(complete["done"])==6
    for f in range(5):
        report=json.loads((audit/f"audits/fold{f}/result.json").read_text())
        assert all(x["passed"] for x in report["comparisons"].values())
    stage=ROOT/"experimental_stage"
    (stage/"model").mkdir(parents=True,exist_ok=False)
    files={}
    def copy(src,dst):
        assert not dst.exists()
        before=sha(src)
        shutil.copy2(src,dst)
        assert sha(dst)==before
        files[str(dst.relative_to(stage))]=dict(sha256=before,bytes=dst.stat().st_size)
    old=Path("experiments/deploy_review_20260905/staged")
    protected=json.loads(Path("experiments/deploy_review_20260905/package_manifest.json").read_text())["files"]
    for name, item in protected.items():
        assert sha(old/name)==item["sha256"], f"Protected stage changed: {name}"
    for name in ("cpsam_v2","seed42.pt","seed77.pt","seed2026.pt","metric35.pt","predbag2_35.pt"):
        copy(old/"model"/name,stage/"model"/name)
    for policy in ("anchor_quality","full_quality"):
        path=Path("experiments/deploy_region_20260907/runs")/policy/"selected.pt"
        audited=json.loads((audit/"audits/all_data/result.json").read_text())["checkpoint_sha256"]
        assert sha(path)==audited[str(path)]
        copy(path,stage/"model"/f"{policy}_seed77.pt")
    copy(ROOT/"source/region_candidate_entrypoint.py",stage/"inference.py")
    copy(old/"inference.py",stage/"base_inference.py")
    for name in ("final_candidate_core.py","fullframe_views.py","native_region_views.py"):
        copy(FROZEN/name,stage/name)
    copy(audit/"source/region_inference_core.py",stage/"region_inference_core.py")
    copy(old/"requirements.txt",stage/"requirements.txt")
    # Deliberately no copied Dockerfile: its COPY list/default config are not valid
    # for this experimental stage. Build configuration requires separate review.
    with np.load("experiments/deploy_review_20260905/inference_predictions.npz",allow_pickle=True) as base:
        target=base["target"].astype(float)
        rows=[]
        for f in range(5):
            with np.load(audit/f"audits/fold{f}/predictions.npz",allow_pickle=True) as data:
                rows.append({k:data[k].copy() for k in data.files})
        merged={k:np.concatenate([r[k] for r in rows]) for k in rows[0]}
        for key in ("indices","ids","target"):
            np.testing.assert_array_equal(merged[key],base[key])
        reports={}
        for policy in ("anchor_quality","full_quality"):
            raw=(5*base["candidate5_bf16"].astype(float)+merged[f"{policy}_bf16"])/6
            fp16=(5*base["candidate5_fp16"].astype(float)+merged[f"{policy}_fp16"])/6
            delta=float(metrics(fp16,target).mean()-metrics(raw,target).mean())
            assert abs(delta)<=.01
            cfg=dict(region_policy=policy,members=["seed42.pt","seed77.pt","seed2026.pt","metric35.pt","predbag2_35.pt",f"{policy}_seed77.pt"],
                calibration=scale_fit(raw,target).tolist(),view_batch=8,experimental_only=True,
                provenance="Fixed equal-weight six members, scalar calibration on historical BF16 OOF; not nested validation. No all-data validation score.")
            path=stage/"model"/f"candidate6_{policy}.json"
            save_json(path,cfg)
            files[str(path.relative_to(stage))]=dict(sha256=sha(path),bytes=path.stat().st_size)
            reports[policy]=dict(raw=float(metrics(raw,target).mean()),fp16_raw=float(metrics(fp16,target).mean()),precision_delta=delta,
                exploratory_crossfit=float(metrics(crossfit_scale(raw,target,base["folds"])[0],target).mean()))
    save_json(ROOT/"stage_manifest.json",dict(files=files,oof=reports,created_at=time.strftime("%FT%T%z"),
        deployment_approved=False,status="Separate experimental integration stage, not submission-ready; no Dockerfile or target-runtime validation"))
    processes={}
    (ROOT/"logs").mkdir(exist_ok=False)
    for gpu,policy in enumerate(("anchor_quality","full_quality")):
        usage=int(subprocess.check_output(["nvidia-smi",f"--id={gpu}","--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True).strip())
        assert usage<100,"GPU occupied"
        cmd=[sys.executable,str(ROOT/"source/prepare_region_candidate.py"),"--worker",policy]
        with (ROOT/"logs"/f"{policy}.log").open("x") as log:
            p=subprocess.Popen(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="4"),stdout=log,stderr=subprocess.STDOUT)
        processes[policy]=p
    save_json(ROOT/"status.json",dict(stage="integration",active={k:p.pid for k,p in processes.items()},updated_at=time.strftime("%FT%T%z")))
    failed={}
    for policy,p in processes.items():
        try:
            rc=p.wait(timeout=1800)
        except subprocess.TimeoutExpired:
            p.terminate()
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill(); p.wait()
            rc=-999
        if rc:
            failed[policy]=rc
    save_json(ROOT/"complete.json",dict(failed=failed,completed_at=time.strftime("%FT%T%z"),deployment_approved=False))


if __name__=="__main__":
    main()
