"""Predeclared two-fold screen and separate three-fold confirmation.

All folds were used in historical development; confirmation here means unused
for THIS recipe-screen decision, not untouched test data or nested validation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from audit_submission_ensembles import metrics, sources, crossfit_scale

METHODS = ("cosine35", "metric35", "predbag2_35", "sam35")
ROOT = Path("experiments/methods_20260904")


def run_path(method, fold, root=ROOT):
    crops = 2 if method == "predbag2_35" else 1
    return root / "runs" / f"{method}_f{fold}_seed42_c256k{crops}_mean_multitask_slidesmooth"


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def load_fold(run):
    if not (run / "result.json").exists():
        raise FileNotFoundError(run / "result.json")
    result = json.loads((run / "result.json").read_text())
    with np.load(run / "best_oof.npz", allow_pickle=True) as d:
        data = {key: d[key].copy() for key in ("indices", "ids", "pred", "target")}
    with np.load(run / "split_indices.npz") as split:
        assert np.array_equal(data["indices"], split["validation"])
        assert not np.intersect1d(data["indices"], split["train"]).size
        data["train"] = split["train"].copy()
    if len(np.unique(data["indices"])) != len(data["indices"]):
        raise ValueError("duplicate validation indices")
    measured = metrics(data["pred"].astype(float), data["target"].astype(float))
    if not np.allclose(measured, result["final_metrics"], atol=0.0001):
        raise ValueError("final D8 result disagrees with saved predictions")
    return data


def aligned(reference, other):
    for key in ("indices", "ids", "target", "train"):
        if not np.array_equal(reference[key], other[key]):
            raise ValueError(f"OOF alignment mismatch: {key}")


def compare(method, folds):
    historical = sources()
    entries = []
    for fold in folds:
        candidate = load_fold(run_path(method, fold))
        old = [load_fold(historical[name][fold]) for name in
               ("fixed_base", "base_seed77", "base_seed2026")]
        for data in old:
            aligned(candidate, data)
        target = candidate["target"].astype(float)
        pred = candidate["pred"].astype(float)
        ensemble = np.mean([d["pred"].astype(float) for d in old], axis=0)
        # Four equally weighted models; no fitted blending weight.
        blend = 0.75 * ensemble + 0.25 * pred
        entries.append(dict(fold=fold, indices=candidate["indices"], ids=candidate["ids"],
            target=target, standalone=pred, seed42=old[0]["pred"].astype(float),
            ensemble=ensemble, blend=blend))
    target = np.concatenate([e["target"] for e in entries])
    values = {name: np.concatenate([e[name] for e in entries]) for name in
              ("standalone", "seed42", "ensemble", "blend")}
    scores = {name: float(metrics(pred, target).mean()) for name, pred in values.items()}
    per_fold = [{"fold": e["fold"], **{name: float(metrics(e[name], e["target"]).mean())
                 for name in values}} for e in entries]
    ds = [e["standalone"] - e["seed42"] for e in per_fold]
    db = [e["blend"] - e["ensemble"] for e in per_fold]
    summary = dict(method=method, n=len(target), folds=list(folds), scores=scores,
        standalone_delta=scores["standalone"] - scores["seed42"],
        blend_delta=scores["blend"] - scores["ensemble"], per_fold=per_fold,
        standalone_fold_deltas=ds, blend_fold_deltas=db)
    return summary, entries, values


def gate(summary):
    # Fixed before launching. Negative delta is better. Allow a small individual
    # fold regression, never select solely from calibration or optimized weights.
    single = summary["standalone_delta"] <= -0.02 and max(summary["standalone_fold_deltas"]) <= 0.05
    blend = summary["blend_delta"] <= -0.01 and max(summary["blend_fold_deltas"]) <= 0.02
    return single or blend


def screen(failed_methods=()):
    summaries = []
    for method in METHODS:
        if method in failed_methods:
            continue
        result, _, _ = compare(method, (0, 1))
        result["passes_screen"] = gate(result)
        summaries.append(result)
    eligible = sorted((x for x in summaries if x["passes_screen"]),
                      key=lambda x: (x["blend_delta"], x["standalone_delta"], x["method"]))
    report = dict(scope="Two-fold exploratory screening on historically reused validation data.",
        gate="standalone delta <= -0.02 and worst fold <= +0.05 OR fixed four-model blend delta <= -0.01 and worst fold <= +0.02; rank by blend delta, retain at most two",
        results=summaries, failed_methods=list(failed_methods),
        promoted=[x["method"] for x in eligible[:2]])
    save(ROOT / "screen.json", report)
    return report


def confirmation(promoted):
    report = dict(scope="Confirm on folds 2/3/4, not used for this screen decision; historical validation reuse and per-fold checkpoint selection still apply. No leaderboard forecast or automatic candidate replacement.",
                  methods={}, incomplete=[])
    for method in promoted:
        if not all((run_path(method, f) / "result.json").exists() for f in range(5)):
            report["incomplete"].append(method)
            continue
        heldout, _, _ = compare(method, (2, 3, 4))
        overall, entries, values = compare(method, range(5))
        target = np.concatenate([e["target"] for e in entries])
        folds = np.concatenate([np.full(len(e["target"]), e["fold"]) for e in entries])
        overall["exploratory_crossfit_calibration"] = {}
        for name in ("ensemble", "blend", "standalone"):
            calibrated, scales = crossfit_scale(values[name], target, folds)
            overall["exploratory_crossfit_calibration"][name] = dict(
                overall=float(metrics(calibrated, target).mean()), scales=scales)
        # Confirmation is a triage flag for human review, not a final-model switch.
        confirmed = (heldout["blend_delta"] < 0 and
            sum(x < 0 for x in heldout["blend_fold_deltas"]) >= 2 and overall["blend_delta"] <= -0.01)
        report["methods"][method] = dict(confirmation=heldout, five_fold=overall,
            blend_confirmation_passed=confirmed)
        np.savez_compressed(ROOT / f"{method}_five_fold_predictions.npz",
            indices=np.concatenate([e["indices"] for e in entries]),
            ids=np.concatenate([e["ids"] for e in entries]), target=target, folds=folds, **values)
    save(ROOT / "confirmation.json", report)
    return report
