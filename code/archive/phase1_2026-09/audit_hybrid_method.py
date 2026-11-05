"""Five-fold comparison for the predeclared metric-loss + prediction-bag hybrid."""
import json
from pathlib import Path

import numpy as np

from audit_method_screen import aligned, load_fold, save, run_path as method_path
from audit_submission_ensembles import crossfit_scale, metrics, sources


ROOT = Path("experiments/hybrid_20260905")
TAG = "metric_predbag2_35"


def run_path(fold):
    return ROOT / "runs" / f"{TAG}_f{fold}_seed42_c256k2_mean_multitask_slidesmooth"


def comparison_gate(methods):
    """Accuracy triage only; never grants deployment or submission approval."""
    candidate = methods["blend"]
    references = [methods[name] for name in ("current4", "current5")]
    return bool(all(candidate["raw_overall"] < ref["raw_overall"] and
        candidate["exploratory_crossfit_overall"] < ref["exploratory_crossfit_overall"] and
        max(np.asarray(candidate["per_fold_raw"]) - np.asarray(ref["per_fold_raw"])) <= 0.02
        for ref in references))


def main():
    historical = sources()
    entries = []
    for fold in range(5):
        hybrid = load_fold(run_path(fold))
        old = [load_fold(historical[name][fold]) for name in
               ("fixed_base", "base_seed77", "base_seed2026")]
        for member in old:
            aligned(hybrid, member)
        metric = load_fold(method_path("metric35", fold))
        bag = load_fold(method_path("predbag2_35", fold))
        aligned(hybrid, metric); aligned(hybrid, bag)
        old_ensemble = np.mean([x["pred"].astype(float) for x in old], axis=0)
        standalone = hybrid["pred"].astype(float)
        entries.append(dict(fold=fold, indices=hybrid["indices"], ids=hybrid["ids"],
            target=hybrid["target"].astype(float), standalone=standalone,
            seed42=old[0]["pred"].astype(float), ensemble=old_ensemble,
            blend=0.75 * old_ensemble + 0.25 * standalone,
            current4=0.75 * old_ensemble + 0.25 * metric["pred"].astype(float),
            current5=0.60 * old_ensemble + 0.20 * metric["pred"].astype(float) + 0.20 * bag["pred"].astype(float)))
    target = np.concatenate([x["target"] for x in entries])
    folds = np.concatenate([np.full(len(x["target"]), x["fold"]) for x in entries])
    values = {name: np.concatenate([x[name] for x in entries]) for name in
              ("standalone", "seed42", "ensemble", "blend", "current4", "current5")}
    report = {"scope": "Five-fold ordinary OOF on historically reused folds; no leaderboard forecast.",
              "method": TAG, "n": len(target), "methods": {}}
    for name, prediction in values.items():
        raw = metrics(prediction, target)
        calibrated, scales = crossfit_scale(prediction, target, folds)
        report["methods"][name] = dict(raw=raw.tolist(), raw_overall=float(raw.mean()),
            per_fold_raw=[float(metrics(prediction[folds == f], target[folds == f]).mean()) for f in range(5)],
            exploratory_crossfit_overall=float(metrics(calibrated, target).mean()), scales=scales)
    report["standalone_delta_vs_seed42"] = (report["methods"]["standalone"]["raw_overall"] -
                                             report["methods"]["seed42"]["raw_overall"])
    report["blend_delta_vs_existing3"] = (report["methods"]["blend"]["raw_overall"] -
                                           report["methods"]["ensemble"]["raw_overall"])
    report["blend_delta_vs_current5"] = report["methods"]["blend"]["raw_overall"] - report["methods"]["current5"]["raw_overall"]
    report["passes_accuracy_triage"] = comparison_gate(report["methods"])
    report["passes_deployment_gate"] = False
    report["gate_note"] = "Revised review criterion, not the original predeclared screen. Compare against BOTH current4/current5; raw and exploratory calibrated improvement, worst fold <= +0.02. Even a pass needs independent validation and target runtime testing. No automatic replacement."
    save(ROOT / "audit_v2.json", report)
    np.savez_compressed(ROOT / "predictions_v2.npz", target=target, folds=folds,
                        indices=np.concatenate([x["indices"] for x in entries]), **values)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
