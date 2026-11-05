#!/usr/bin/env python3
"""Exploratory out-of-fold calibration diagnostic, never deployment selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression

from audit_ensemble_20260923 import FIRST_FIVE, SCALE, mape, path_for, read_npz, summarize, target_tails


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", type=Path, default=Path("experiments"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    parts = []
    for fold in range(5):
        arrays = [read_npz(path_for(args.experiments, name, fold)) for name in FIRST_FIVE]
        ref = arrays[0]
        for other in arrays[1:]:
            for key in ("indices", "ids", "target"):
                np.testing.assert_array_equal(ref[key], other[key])
        parts.append((np.full(len(ref["pred"]), fold), ref["target"].astype(float),
                      np.mean([a["pred"].astype(float) for a in arrays], axis=0)))
    fold = np.concatenate([x[0] for x in parts]); y = np.concatenate([x[1] for x in parts]); p = np.concatenate([x[2] for x in parts])
    iso = np.zeros_like(p)
    power = np.zeros_like(p)
    scalar = np.zeros_like(p)
    parameters = {}
    for outer in range(5):
        parameters[str(outer)] = {}
        for j, metric in enumerate(("CD", "CV", "HEX")):
            train = (fold != outer) & (y[:, j] > 0) & np.isfinite(y[:, j])
            test = fold == outer
            x, target = p[train, j], y[train, j]
            if (x <= 0).any():
                raise ValueError("positive prediction required")
            model = IsotonicRegression(increasing=True, out_of_bounds="clip")
            model.fit(x, target, sample_weight=1 / target)
            iso[test, j] = model.predict(p[test, j])
            center = np.log(x).mean()
            def transformed(theta, raw):
                return raw * np.exp(theta[0] + theta[1] * (np.log(raw) - center))
            def objective(theta):
                q = transformed(theta, x)
                return np.mean(np.abs(q - target) / target)
            fit = minimize(objective, [0., 0.], method="Powell",
                           bounds=[(-.3, .3), (-1., 1.)], options={"maxiter": 250, "xtol": 1e-5, "ftol": 1e-7})
            if not fit.success or not np.isfinite(fit.x).all():
                raise RuntimeError(f"power fit failed: fold={outer}, metric={metric}: {fit.message}")
            power[test, j] = transformed(fit.x, p[test, j])
            ratios, weights = target / x, x / target
            order = np.argsort(ratios)
            factor = ratios[order][np.searchsorted(np.cumsum(weights[order]), weights.sum() / 2)]
            scalar[test, j] = p[test, j] * factor
            parameters[str(outer)][metric] = {"power_intercept": float(fit.x[0]),
                                               "power_slope_delta": float(fit.x[1]),
                                               "scalar": float(factor),
                                               "isotonic_knots": int(len(model.X_thresholds_))}
    methods = {"raw_five": p, "historical_fixed_scale": p * SCALE,
               "crossfit_scalar": scalar, "crossfit_power": power, "crossfit_isotonic": iso}
    report = {"scope": "Exploratory meta cross-fitting on reused OOF base models. Base training is NOT nested relative to outer calibration folds; this is not independent validation or a deployment recommendation.",
              "n": len(y), "methods": {}, "parameters": parameters}
    for name, pred in methods.items():
        report["methods"][name] = {**summarize(pred, y, fold), "target_tails": target_tails(pred, y)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v["overall"] for k, v in report["methods"].items()}, indent=2))


if __name__ == "__main__":
    main()
