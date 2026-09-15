"""Read-only paired two-fold specialist audit; fixed per-target 50/50 blends."""
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path("experiments/specialize_screen_20260908")
OLD = Path("experiments/region_search_20260905/source/code")
sys.path.insert(0, str(OLD))
import audit_region_search as historical
from audit_submission_ensembles import metrics, sources
from audit_method_screen import run_path as method_path


def main():
    output = ROOT / "screen_audit.json"
    if output.exists():
        raise FileExistsError(output)
    complete = json.loads((ROOT / "complete.json").read_text())
    assert not complete["failed"] and sorted(complete["done"]) == [0,1]
    historical.run_path = lambda recipe, fold: Path("experiments/region_seedcheck_20260906/runs") / (
        f"full_quality_f{fold}_seed77_c256k2_mean_multitask_slidesmooth")
    _, old = historical.compare("full_quality", (0,1))
    hashes = {}
    def record(path):
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for f in (0,1):
        folders = [sources()[key][f] for key in ("fixed_base", "base_seed77", "base_seed2026")]
        folders += [method_path("metric35", f), method_path("predbag2_35", f), historical.run_path("full_quality", f)]
        for folder in folders:
            for name in ("best_oof.npz", "result.json", "split_indices.npz"):
                record(folder / name)
    predictions = {}
    for branch in ("multitask", "cd", "cv", "hex"):
        entries = []
        for f in (0,1):
            folder = ROOT / "runs" / f"fold{f}"
            with np.load(folder / branch / "oof.npz") as data:
                mask = old["folds"] == f
                for key in ("indices", "ids", "target"):
                    np.testing.assert_array_equal(data[key], old[key][mask])
                entries.append(data["pred"].astype(float))
            with np.load(folder / "split_indices.npz") as split, np.load(historical.run_path("full_quality", f) / "split_indices.npz") as ref:
                np.testing.assert_array_equal(split["train"], ref["train"])
                np.testing.assert_array_equal(split["validation"], ref["validation"])
            record(folder / branch / "oof.npz"); record(folder / branch / "result.json")
        predictions[branch] = np.concatenate(entries)
    base, target, folds = old["add6"], old["target"], old["folds"]
    control = predictions["multitask"]
    report = dict(created_at=time.strftime("%FT%T%z"), deployment_approved=False,
        n=len(target), current6_metrics=metrics(base,target).tolist(),
        matched_multitask_metrics=metrics(control,target).tolist(), specialists={},
        scope="Historically reused slide-group OOF. Fixed terminal weights; not independent test. Confirm remaining folds before adoption.")
    assert len(target) == 3582 and len(np.unique(old["indices"])) == 3582
    for j, name in enumerate(("cd", "cv", "hex")):
        p = predictions[name]
        blend = base.copy(); blend[:,j] = (base[:,j]+p[:,j])/2
        control_blend = base.copy(); control_blend[:,j] = (base[:,j]+control[:,j])/2
        delta_target = float(metrics(p,target)[j]-metrics(control,target)[j])
        delta_overall = float((metrics(blend,target)-metrics(base,target)).mean())
        per_fold = []
        for f in (0,1):
            m = folds == f
            per_fold.append(dict(fold=f,
                target_delta=float(metrics(p[m],target[m])[j]-metrics(control[m],target[m])[j]),
                blend_overall_delta=float((metrics(blend[m],target[m])-metrics(base[m],target[m])).mean()),
                blend_vs_control_overall_delta=float((metrics(blend[m],target[m])-metrics(control_blend[m],target[m])).mean())))
        passed = (delta_target <= -.05 and delta_overall <= -.01 and
            all(r["target_delta"] < 0 and r["blend_overall_delta"] < 0 for r in per_fold))
        report["specialists"][name] = dict(target_metric=float(metrics(p,target)[j]),
            target_delta_vs_matched_control=delta_target, fixed_blend_metrics=metrics(blend,target).tolist(),
            fixed_blend_overall=float(metrics(blend,target).mean()), blend_overall_delta=delta_overall,
            blend_vs_control_overall_delta=float((metrics(blend,target)-metrics(control_blend,target)).mean()),
            per_fold=per_fold, passes_screen=bool(passed))
    report["eligible_for_confirmation"] = [name for name,r in report["specialists"].items() if r["passes_screen"]]
    for path in (Path(__file__), OLD / "audit_region_search.py", OLD / "audit_method_screen.py", OLD / "audit_submission_ensembles.py", ROOT / "manifest.json"):
        record(path)
    report["input_sha256"] = hashes
    with output.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps({k:v for k,v in report.items() if k != "input_sha256"}, indent=2))


if __name__ == "__main__":
    main()
