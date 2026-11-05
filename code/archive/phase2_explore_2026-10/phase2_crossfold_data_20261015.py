#!/usr/bin/env python3
"""Read-only cross-fold inventory; emit one new aggregate JSON, never IDs/images.

Requires NumPy only. Run on the PVC with --verify-pixels to compare every current
full-frame pixel SHA256 against historical fingerprints. Without that flag,
pixel overlap checks use historical positional evidence and are labelled so.
No labels, splits, models, historic files or directories are changed.

The discovery scope is deliberately shallow: experiments/*/preflight/fold*.npz
and a supplied --split path. No model/weight or broad recursive traversal.
Exit 0 means the shared data checks passed; inspect each split's eligibility.
Exit 2 means a shared data check failed; exit 3 means required evidence failed.
Neither a zero exit nor a disjoint split establishes patient independence.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np

METRICS = ("CD", "CV", "HEX")
BASELINE = "experiments/overnight_target_20260912/preflight/fold3.npz"
FOLD4 = "experiments/target_fold4_20260912/preflight/fold4.npz"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def stats(y):
    result = {}
    for j, metric in enumerate(METRICS):
        values = y[:, j]
        finite = values[np.isfinite(values)]
        result[metric] = dict(n=len(values), finite=int(np.isfinite(values).sum()),
            positive=int((values > 0).sum()), zeros=int((values == 0).sum()),
            negative=int((values < 0).sum()),
            min=float(finite.min()) if len(finite) else None,
            max=float(finite.max()) if len(finite) else None,
            mean=float(finite.mean()) if len(finite) else None)
    return result


def connected_groups(slides, fingerprints):
    parent = {str(s): str(s) for s in slides}

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    seen = {}
    for slide, digest in zip(slides, fingerprints):
        slide, digest = str(slide), str(digest)
        if digest in seen:
            parent[find(slide)] = find(seen[digest])
        else:
            seen[digest] = slide
    return np.asarray([find(str(s)) for s in slides])


class Audit:
    def __init__(self, root):
        self.root = root
        self.report = dict(created_utc=time.strftime("%FT%TZ", time.gmtime()),
            source_sha256=sha(__file__), aggregate_only=True, inputs={}, checks=[],
            errors=[], splits={}, unresolved=[
                "Patient identity has no verified mapping; slide names are not patient IDs.",
                "Near-duplicate similarity is not exhaustively audited.",
                "Historically reused folds are not untouched test data.",
                "Official hidden-platform zero-target policy is not established by the public local baseline."])
        self.cache = {}

    def path(self, value):
        value = Path(value)
        return value if value.is_absolute() else self.root / value

    def name(self, value):
        value = self.path(value)
        try:
            return str(value.relative_to(self.root))
        except ValueError:
            return str(value)

    def identify(self, value):
        value = self.path(value)
        key = self.name(value)
        if key not in self.report["inputs"]:
            self.report["inputs"][key] = dict(bytes=value.stat().st_size, sha256=sha(value))
        return value

    def npz(self, value):
        value = self.identify(value)
        # These are existing trusted research metadata, including object-string IDs.
        with np.load(value, allow_pickle=True) as data:
            result = {k: data[k].copy() for k in data.files}
        self.report["inputs"][self.name(value)]["arrays"] = {
            k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in result.items()}
        return result

    def read_json(self, value):
        value = self.identify(value)
        key = self.name(value)
        if key not in self.cache:
            self.cache[key] = json.loads(value.read_text())
        return self.cache[key]

    def check(self, name, passed, **details):
        self.report["checks"].append(dict(name=name, passed=bool(passed), **details))


def load_data(a, verify_pixels):
    native = a.npz("datasets/clear_ec_official/cache_native_meta_512.npz")
    full = a.npz("datasets/clear_ec_fullframe_20260904/fullframe_meta.npz")
    ids, y = native["ids"].astype(str), native["Y"]
    n = len(ids)
    if ids.ndim != 1 or y.shape != (n, 3):
        raise ValueError("Unexpected native metadata shape")
    a.check("unique_ids", len(np.unique(ids)) == n)
    a.check("finite_nonnegative_targets", np.isfinite(y).all() and (y >= 0).all())
    a.check("full_native_id_order", np.array_equal(ids, full["ids"].astype(str)))
    a.check("full_native_labels", "Y" in full and np.array_equal(y, full["Y"]))
    label_path = a.identify("datasets/clear_ec_official/final_train_ids.csv")
    with label_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames or [], list(reader)
    if not {"ID", "slide_id", *METRICS}.issubset(fields):
        raise ValueError("Source CSV lacks required ID/slide/target fields")
    by_id = defaultdict(list)
    for row in rows:
        by_id[Path(row["ID"]).stem].append(row)
    keys = [Path(value).stem for value in ids]
    a.check("unique_normalized_cache_ids", len(set(keys)) == n)
    valid_mapping = all(len(by_id[key]) == 1 for key in keys)
    a.check("one_source_csv_row_per_cache_id", valid_mapping)
    if not valid_mapping:
        raise ValueError("Ambiguous or missing source CSV rows")
    aligned = [by_id[key][0] for key in keys]
    source_y = np.asarray([[float(row[m]) for m in METRICS] for row in aligned], dtype=y.dtype)
    a.check("raw_source_labels_unchanged", np.array_equal(y, source_y),
            mismatch_entries=int(np.count_nonzero(y != source_y)))
    slides = np.asarray([str(row["slide_id"]).strip() for row in aligned])
    a.check("all_slide_groups_present", all(slides))
    evidence = a.npz("experiments/ssl_screen_20260911/preflight/pixel_fingerprints.npz")
    hashes = evidence["sha256"].astype(str)
    if hashes.shape != (n,):
        raise ValueError("Pixel fingerprint shape mismatch")
    for key in ("ids", "ID"):
        if key in evidence:
            a.check("fingerprint_id_order", np.array_equal(ids, evidence[key].astype(str)))
    manifest = a.read_json("datasets/clear_ec_fullframe_20260904/fullframe_manifest.json")
    a.check("raw_cache_shape", manifest.get("cache_shape") == [n, 972, 1296])
    raw_path = a.path("datasets/clear_ec_fullframe_20260904/img_full_raw.dat")
    expected_bytes = n * 972 * 1296
    a.check("raw_cache_byte_count", raw_path.stat().st_size == expected_bytes)
    pixels = dict(verified=False, fingerprint_evidence="historical positional SHA256", n=n)
    if verify_pixels:
        if raw_path.stat().st_size != expected_bytes:
            raise ValueError("Cannot verify malformed raw pixel cache")
        raw = np.memmap(raw_path, mode="r", dtype=np.uint8, shape=(n, 972, 1296))
        started, mismatches = time.monotonic(), 0
        total_digest = hashlib.sha256()
        current = []
        for index in range(n):
            payload = raw[index].tobytes()
            total_digest.update(payload)
            digest = hashlib.sha256(payload).hexdigest()
            current.append(digest)
            mismatches += int(digest != hashes[index])
        a.check("all_current_pixels_match_fingerprints", mismatches == 0, mismatch_rows=mismatches)
        hashes = np.asarray(current)
        pixels = dict(verified=True, fingerprint_evidence="current full-frame pixels SHA256",
            n=n, mismatch_rows=mismatches, seconds=time.monotonic() - started,
            whole_cache_sha256=total_digest.hexdigest())
    else:
        a.report["unresolved"].append("Current pixel hashes not rechecked; use --verify-pixels before training.")
    a.report["pixels"] = pixels
    a.report["dataset"] = dict(images=n, labels=stats(y), label_order=list(METRICS),
        label_transform="none", slide_groups=len(np.unique(slides)),
        ids_sha256=array_sha(ids), labels_sha256=array_sha(y),
        full_cache_provenance={k: manifest.get(k) for k in (
            "cache_shape", "archive_sha256", "central_pixels_exact_match", "archive_labels_match_cache")})
    return ids, y, slides, hashes, connected_groups(slides, hashes)


def valid_indices(indices, n):
    return (indices.ndim == 1 and np.issubdtype(indices.dtype, np.integer)
        and len(np.unique(indices)) == len(indices)
        and (not len(indices) or (indices.min() >= 0 and indices.max() < n)))


def overlap(left, right, groups):
    shared = np.intersect1d(groups[left], groups[right])
    return dict(groups=len(shared), left_rows=int(np.isin(groups[left], shared).sum()),
                right_rows=int(np.isin(groups[right], shared).sum()))


def inspect_split(a, path, ids, y, slides, hashes, connected):
    data = a.npz(path)
    result = dict(path=a.name(path), split_sha256=sha(path), checks=[],
                  historical_validation_reuse=True, patient_independence="unresolved")

    def check(name, passed, **details):
        result["checks"].append(dict(name=name, passed=bool(passed), **details))

    for key in ("train", "validation"):
        if key not in data:
            raise ValueError("Split missing train/validation")
    for key in ("train", "validation", "primary_validation", "historical_train"):
        if key not in data:
            continue
        values = data[key]
        check(key + "_valid_unique_indices", valid_indices(values, len(ids)))
        if not valid_indices(values, len(ids)):
            result["eligible_positive_target_candidate"] = False
            return result, None
        result[key] = dict(n=len(values), labels=stats(y[values]),
            indices_sha256=array_sha(values), ids_sha256=array_sha(ids[values]),
            target_sha256=array_sha(y[values]))
    tr, va = data["train"], data["validation"]
    check("nonempty_partitions", len(tr) > 0 and len(va) > 0)
    check("index_disjoint", not np.intersect1d(tr, va).size)
    check("id_disjoint", not np.intersect1d(ids[tr], ids[va]).size)
    result["overlap"] = {}
    for label, groups in (("slide", slides), ("exact_pixel", hashes), ("slide_exact_connected", connected)):
        value = overlap(tr, va, groups)
        result["overlap"][label] = value
        check(label + "_disjoint", value["groups"] == 0)
    assigned = np.union1d(tr, va)
    unassigned = np.setdiff1d(np.arange(len(ids)), assigned)
    result["unassigned"] = dict(n=len(unassigned), labels=stats(y[unassigned]))
    if "historical_train" in data and "primary_validation" in data:
        historical, primary = data["historical_train"], data["primary_validation"]
        leaking_groups = np.intersect1d(connected[historical], connected[va])
        expected_train = historical[~np.isin(connected[historical], leaking_groups)]
        expected_primary = va[~np.isin(connected[va], leaking_groups)]
        check("clean_train_reconstructs", np.array_equal(tr, expected_train))
        check("legacy_primary_reconstructs", np.array_equal(primary, expected_primary))
        check("primary_is_validation_subset", np.isin(primary, va).all())
        result["legacy_cleaning"] = dict(
            historic_overlap=overlap(historical, va, connected),
            removed_train_rows=len(historical) - len(tr),
            excluded_legacy_primary_validation_rows=len(va) - len(primary),
            explanation="clean_train removes historical train groups shared with any validation row; legacy primary excludes the corresponding validation rows for comparison to uncleaned historical models",
            evaluation_proposal="If all checks pass, freshly trained SSL/regression use clean_train only, all validation stays held out and is primary; legacy primary subset is a secondary historical comparison.")
    manifest_path = path.parent.parent / "manifest.json"
    result["provenance"] = dict(manifest_path=a.name(manifest_path), exists=manifest_path.is_file())
    matched_records = []
    if manifest_path.is_file():
        manifest = a.read_json(manifest_path)
        for fold, record in manifest.get("folds", {}).items():
            if not isinstance(record, dict) or not record.get("split_path"):
                continue
            if a.path(record["split_path"]).resolve() != path.resolve():
                continue
            matched_records.append(str(fold))
            check("manifest_split_sha_" + str(fold), record.get("split_sha256") == sha(path))
            for key, expected in (("train_rows", len(tr)), ("validation_rows", len(va))):
                if key in record:
                    check("manifest_" + key + "_" + str(fold), record[key] == expected)
            if "primary_rows" in record and "primary_validation" in data:
                check("manifest_primary_rows_" + str(fold), record["primary_rows"] == len(data["primary_validation"]))
        historical_input_checks = []
        for relative, expected in manifest.get("input_sha256", {}).items():
            observed = a.report["inputs"].get(a.name(relative), {}).get("sha256")
            if observed is not None:
                passed = observed == expected
                historical_input_checks.append(dict(path=a.name(relative), matched=passed))
                check("manifest_input_sha:" + a.name(relative), passed)
        result["provenance"].update(matched_manifest_folds=matched_records,
            input_hash_checks=historical_input_checks,
            historic_selection=manifest.get("selection"),
            historic_limitations=manifest.get("limitations"))
        reference_path = path.parent / "reference.npz"
        if reference_path.is_file() and matched_records:
            ref = a.npz(reference_path)
            for fold in matched_records:
                if not {"folds", "ids", "indices", "target"}.issubset(ref):
                    check("reference_schema", False)
                    break
                selected = ref["folds"] == int(fold)
                check("reference_ids_" + fold, np.array_equal(ref["ids"][selected].astype(str), ids[va]))
                check("reference_targets_" + fold, np.array_equal(ref["target"][selected], y[va]))
                check("reference_indices_" + fold, np.array_equal(ref["indices"][selected], va))
                if "primary" in ref and "primary_validation" in data:
                    check("reference_primary_" + fold, np.array_equal(
                        va[ref["primary"][selected].astype(bool)], data["primary_validation"]))
            expected_reference_sha = manifest.get("input_sha256", {}).get(a.name(reference_path))
            if expected_reference_sha:
                check("reference_manifest_sha", sha(reference_path) == expected_reference_sha)
    result["positive_targets_train_and_validation"] = bool(
        np.isfinite(y[tr]).all() and np.isfinite(y[va]).all() and
        (y[tr] > 0).all() and (y[va] > 0).all())
    result["eligible_positive_target_candidate"] = bool(matched_records and
        result["positive_targets_train_and_validation"] and all(c["passed"] for c in result["checks"]))
    result["eligibility_caveat"] = "Also requires shared data checks and current pixel verification; does not resolve patient identity/near-duplicates or historical tuning reuse."
    return result, data


def run(a, args):
    ids, y, slides, hashes, groups = load_data(a, args.verify_pixels)
    paths = set(a.root.glob("experiments/*/preflight/fold*.npz"))
    paths.update(a.path(value) for value in args.split)
    paths = sorted(p for p in paths if not p.name.startswith("._"))
    a.report["inventory_scope"] = "Only experiments/*/preflight/fold*.npz and explicit --split; no recursive model/weight traversal"
    a.report["inventory_count"] = len(paths)
    loaded = {}
    for path in paths:
        key = a.name(path)
        try:
            result, data = inspect_split(a, path, ids, y, slides, hashes, groups)
            a.report["splits"][key] = result
            if data is not None:
                loaded[key] = data
        except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            a.report["splits"][key] = dict(error=type(exc).__name__ + ": " + str(exc),
                eligible_positive_target_candidate=False)
    for key in (BASELINE, FOLD4):
        a.check("required_split_present:" + key, key in loaded)
    if BASELINE in loaded and FOLD4 in loaded:
        old, new = loaded[BASELINE], loaded[FOLD4]
        a.report["fold3_ssl_leakage_if_reused_for_fold4"] = dict(
            fold3_train_vs_fold4_validation_index_overlap=int(np.intersect1d(old["train"], new["validation"]).size),
            fold3_train_vs_fold4_validation_connected_overlap=overlap(old["train"], new["validation"], groups),
            rule="Train fold4 SSL from the external pretrained initialization using fold4 clean_train only; never reuse fold3-adapted encoder for independent cross-fold confirmation.")
    a.report["zero_policy_evidence"] = dict(
        public_formula_url="https://clear-ec.grand-challenge.org/",
        public_local_baseline_url="https://github.com/CLEAR-EC/CLEAR-EC/blob/main/code/src/utils/evaluate.py",
        observed_local_baseline="GT zero/missing becomes per-metric NaN; per-case errors round to one decimal then pandas mean(skipna=True)",
        hidden_platform_evaluator_verified=False,
        action="Keep all positive fold4 validation rows in the unchanged unsmoothed MAPE protocol; do not silently replace/exclude zeros on other folds.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, required=True, help="New JSON path; parent directory must exist")
    parser.add_argument("--verify-pixels", action="store_true")
    parser.add_argument("--split", action="append", default=[], help="Additional existing split path; repeatable")
    args = parser.parse_args()
    a = Audit(args.root.resolve())
    out = a.path(args.out)
    if out.exists() or not out.parent.is_dir():
        parser.error("Output must be a new file in an existing directory")
    try:
        run(a, args)
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        a.report["errors"].append(type(exc).__name__ + ": " + str(exc))
    failed = [c["name"] for c in a.report["checks"] if not c["passed"]]
    a.report["failed_shared_checks"] = failed
    a.report["status"] = "incomplete" if a.report["errors"] else "failed" if failed else "shared_checks_passed"
    a.report["training_started"] = False
    with out.open("x") as stream:
        json.dump(a.report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status=a.report["status"], output=str(out),
        inventory_count=a.report.get("inventory_count", 0), failed_shared_checks=failed,
        errors=len(a.report["errors"]), pixels_verified=a.report.get("pixels", {}).get("verified", False))))
    return 3 if a.report["errors"] else 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
