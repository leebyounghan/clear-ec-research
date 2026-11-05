"""Read completed, frozen regional OOF runs; fixed-weight, predeclared audit.

Run on the pod from repository root. Writes a new report exclusively; never
changes training, predictions, staged models, or blending weights.
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path("experiments/region_seedconfirm_20260906")
FROZEN = Path("experiments/region_search_20260905/source/code")
sys.path.insert(0, str(FROZEN))
import audit_region_search as audit
from audit_method_screen import aligned, load_fold
from audit_submission_ensembles import metrics


def path_for(recipe, fold, seed):
    root = ("region_search_20260905" if seed == 42 else
            "region_seedcheck_20260906" if fold < 2 else
            "region_seedconfirm_20260906")
    return Path("experiments") / root / "runs" / (
        f"{recipe}_f{fold}_seed{seed}_c256k2_mean_multitask_slidesmooth")


def summarize(data, pred):
    target, folds, base = data["target"], data["folds"], data["current5"]
    value = metrics(pred, target)
    base_value = metrics(base, target)
    per_fold = [float((metrics(pred[folds == f], target[folds == f]) -
                       metrics(base[folds == f], target[folds == f])).mean())
                for f in range(5)]
    confirm = folds >= 2
    return dict(raw=float(value.mean()), cd_cv_hex=value.tolist(),
                delta_cd_cv_hex=(value - base_value).tolist(),
                delta=float((value - base_value).mean()),
                confirmation_delta=float((metrics(pred[confirm], target[confirm]) -
                                          metrics(base[confirm], target[confirm])).mean()),
                fold_deltas=per_fold, worst_fold_delta=max(per_fold))


def main():
    complete = json.loads((ROOT / "complete.json").read_text())
    assert len(complete["done"]) == 12 and not complete["failed"]
    output = ROOT / "manual_fivefold_audit_20260907.json"
    if output.exists():
        raise FileExistsError(output)
    report = dict(created_at=datetime.now(timezone.utc).isoformat(),
                  completion=complete, deployment_approved=False,
                  scope="Historically reused slide-group OOF with validation checkpoint selection; not independent test or leaderboard forecast.",
                  gate="Each new seed: confirmation delta <= -0.01, five-fold delta <= -0.01, all five fold deltas < 0.",
                  policies={}, input_sha256={})
    for recipe in ("anchor_quality", "full_quality"):
        datasets, results = {}, {}
        for seed in (42, 77, 2026):
            for fold in range(5):
                run = path_for(recipe, fold, seed)
                loaded = load_fold(run)
                aligned(load_fold(path_for(recipe, fold, 42)), loaded)
                assert (loaded["pred"] > 0).all()
                assert len(loaded["indices"]) == (1790 if fold == 4 else 1791)
                for name in ("result.json", "best_oof.npz", "split_indices.npz"):
                    p = run / name
                    report["input_sha256"][str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
            audit.run_path = lambda r, f, s=seed: path_for(r, f, s)
            original, data = audit.compare(recipe, range(5))
            assert len(np.unique(data["indices"])) == 8954
            if datasets:
                for key in ("indices", "ids", "target", "folds", "current5"):
                    assert np.array_equal(data[key], datasets[42][key]), key
            datasets[seed] = data
            result = summarize(data, data["add6"])
            result["standalone_raw"] = original["scores"]["standalone"]
            result["exploratory_calibrated_raw"] = original["exploratory_crossfit"]["add6"]
            result["gate_passed"] = bool(result["confirmation_delta"] <= -.01 and
                                         result["delta"] <= -.01 and
                                         max(result["fold_deltas"]) < 0)
            result["delta_vs_same_policy_seed42"] = result["raw"] - (results[42]["raw"] if results else result["raw"])
            results[seed] = result
        regional_mean = np.mean([datasets[s]["standalone"] for s in (42, 77, 2026)], axis=0)
        average = summarize(datasets[42], (5 * datasets[42]["current5"] + regional_mean) / 6)
        average["actual_model_count"] = 8
        average["delta_vs_same_policy_seed42"] = average["raw"] - results[42]["raw"]
        report["policies"][recipe] = dict(seeds=results, fixed_three_seed_regional_mean=average,
            both_new_seeds_pass=all(results[s]["gate_passed"] for s in (77, 2026)))
        report["current5"] = summarize(datasets[42], datasets[42]["current5"])
    report["audit_source"] = SOURCE_TEXT if "SOURCE_TEXT" in globals() else Path(__file__).read_text()
    with output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    compact = {k: v for k, v in report.items() if k not in ("input_sha256", "audit_source")}
    print(json.dumps(compact, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
