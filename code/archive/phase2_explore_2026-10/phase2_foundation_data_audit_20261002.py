#!/usr/bin/env python3
"""CPU-only, read-only foundation data audit; writes ONE new aggregate JSON.

Run from the research checkout (or pass --root). Requires only NumPy. This does
not infer patient identities, repair labels, create splits, or select examples
for training. Existing local NPZ metadata are trusted inputs (some contain
object-string ID arrays). Default pixel verification reads every full image and
its central 512 cache, without modifying either. --skip-pixels explicitly lowers
the assurance level; historical fingerprints then remain unverified evidence.

Example:
  python code/phase2_foundation_data_audit_20261002.py \
    --out experiments/phase2_foundation_20261002/data_audit.json

Exit 0: implemented consistency checks passed (policy gaps may remain).
Exit 2: a consistency check failed. Exit 3: required evidence missing/unreadable.
Neither exit 0 nor a clean split establishes an independent Phase II test.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np

METRICS = ("CD", "CV", "HEX")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8 << 20), b""):
            h.update(part)
    return h.hexdigest()


def clean_id(value):
    return Path(str(value)).stem


def finite_stats(labels):
    labels = np.asarray(labels)
    result = {}
    for j, metric in enumerate(METRICS):
        x = labels[:, j]
        good = x[np.isfinite(x)]
        result[metric] = dict(n=len(x), finite=int(np.isfinite(x).sum()),
                             zeros=int((x == 0).sum()), negative=int((x < 0).sum()),
                             min=float(good.min()) if len(good) else None,
                             max=float(good.max()) if len(good) else None,
                             mean=float(good.mean()) if len(good) else None)
    return result


class Audit:
    def __init__(self, root):
        self.root = root
        self.report = dict(created_utc=time.strftime("%FT%TZ", time.gmtime()),
                           source_sha256=sha(__file__), inputs={}, checks=[],
                           unresolved=[], errors=[], read_only_inputs=True,
                           models_or_labels_written=False)

    def path(self, value):
        path = Path(value)
        return path if path.is_absolute() else self.root / path

    def identify(self, path, hash_file=True):
        path = self.path(path)
        self.report["inputs"][str(path)] = dict(bytes=path.stat().st_size,
                                               sha256=sha(path) if hash_file else None)
        return path

    def check(self, name, passed, **details):
        self.report["checks"].append(dict(name=name, passed=bool(passed), **details))

    def json(self, path):
        return json.loads(self.identify(path).read_text())

    def npz(self, path):
        path = self.identify(path)
        with np.load(path, allow_pickle=True) as z:
            values = {k: z[k].copy() for k in z.files}
        self.report["inputs"][str(path)]["arrays"] = {
            k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in values.items()}
        return values

    def required(self, data, keys, name):
        missing = sorted(set(keys) - set(data))
        if missing:
            raise ValueError(f"{name}: missing required keys {missing}; found {sorted(data)}")


def group_components(slides, hashes):
    parents = {s: s for s in slides}

    def find(s):
        while parents[s] != s:
            parents[s] = parents[parents[s]]
            s = parents[s]
        return s

    seen = {}
    for slide, digest in zip(slides, hashes):
        if digest in seen:
            parents[find(slide)] = find(seen[digest])
        else:
            seen[digest] = slide
    return np.array([find(s) for s in slides])


def audit_pixels(a, n, full_manifest, historical_hashes, skip):
    shape = full_manifest.get("cache_shape")
    if shape != [n, 972, 1296]:
        a.check("full_cache_shape", False, found=shape, expected=[n, 972, 1296])
        return None
    a.check("full_cache_shape", True)
    if skip:
        a.report["pixels"] = dict(verified=False, reason="--skip-pixels",
                                  fingerprint_provenance="historical positional evidence only")
        a.report["unresolved"].append("Current pixels and fingerprint order were not reverified.")
        return historical_hashes
    full_path = a.identify("datasets/clear_ec_fullframe_20260904/img_full_raw.dat", False)
    central_path = a.identify("datasets/clear_ec_official/img_native_512.dat", False)
    expected_bytes = int(np.prod(shape))
    a.check("full_cache_bytes", full_path.stat().st_size == expected_bytes)
    a.check("central_cache_bytes", central_path.stat().st_size == n * 512 * 512)
    if full_path.stat().st_size != expected_bytes or central_path.stat().st_size != n * 512 * 512:
        return None
    full = np.memmap(full_path, mode="r", dtype=np.uint8, shape=tuple(shape))
    central = np.memmap(central_path, mode="r", dtype=np.uint8, shape=(n, 512, 512))
    hashes, mismatch, start = [], 0, time.monotonic()
    full_digest, central_digest = hashlib.sha256(), hashlib.sha256()
    row, col = (shape[1] - 512) // 2, (shape[2] - 512) // 2
    for i in range(n):
        raw = full[i].tobytes()
        full_digest.update(raw)
        central_digest.update(central[i].tobytes())
        hashes.append(hashlib.sha256(raw).hexdigest())
        mismatch += int(not np.array_equal(full[i, row:row + 512, col:col + 512], central[i]))
    hashes = np.array(hashes)
    a.report["inputs"][str(full_path)]["sha256"] = full_digest.hexdigest()
    a.report["inputs"][str(central_path)]["sha256"] = central_digest.hexdigest()
    a.check("central_pixels_equal_full_center_all_images", mismatch == 0, mismatch_rows=mismatch)
    if historical_hashes is not None:
        a.check("historical_fingerprints_equal_current_pixels", np.array_equal(hashes, historical_hashes))
    a.report["pixels"] = dict(verified=True, images=n, seconds=time.monotonic() - start,
                              central_crop_origin=[row, col], dtype="uint8")
    return hashes


def audit_fold(a, fold, prior, ids, labels, slides, hashes, connected):
    path = a.path(prior["split_path"])
    split = a.npz(path)
    a.required(split, ("train", "validation"), f"fold{fold}")
    a.check(f"fold{fold}_split_sha", sha(path) == prior.get("split_sha256"))
    tr, va = split["train"], split["validation"]
    valid = all(x.ndim == 1 and np.issubdtype(x.dtype, np.integer) and
                (not len(x) or (x.min() >= 0 and x.max() < len(ids))) for x in (tr, va))
    a.check(f"fold{fold}_valid_indices", valid)
    if not valid:
        return {"valid_indices": False}
    for name, values in (("train", tr), ("validation", va)):
        a.check(f"fold{fold}_{name}_unique", len(np.unique(values)) == len(values))
        if prior.get(f"{name}_rows") is not None:
            a.check(f"fold{fold}_{name}_count", len(values) == prior[f"{name}_rows"])
    a.check(f"fold{fold}_index_disjoint", not np.intersect1d(tr, va).size)
    result = dict(train_n=len(tr), validation_n=len(va),
                  train_labels=finite_stats(labels[tr]), validation_labels=finite_stats(labels[va]),
                  historic_validation_reuse=True, patient_independence="unresolved")
    if slides is not None:
        overlap = set(slides[tr]) & set(slides[va])
        a.check(f"fold{fold}_slide_disjoint", not overlap, overlap_groups=len(overlap))
        result["slide_overlap_groups"] = len(overlap)
    for name, grouping in (("exact_pixels", hashes), ("slide_exact_connected", connected)):
        if grouping is not None:
            overlap = set(grouping[tr]) & set(grouping[va])
            a.check(f"fold{fold}_{name}_disjoint", not overlap, overlap_groups=len(overlap))
            result[f"{name}_overlap_groups"] = len(overlap)
    reference = prior.get("reference")
    if not reference and prior.get("reference_root"):
        reference = str(Path(prior["reference_root"]) / "preflight/reference.npz")
    if reference:
        ref = a.npz(reference)
        a.required(ref, ("folds", "indices", "ids", "target", "primary"), "reference")
        mask = ref["folds"] == int(fold)
        for name, expected in (("indices", va), ("ids", ids[va]), ("target", labels[va])):
            found = ref[name][mask].astype(str) if name == "ids" else ref[name][mask]
            a.check(f"fold{fold}_reference_{name}_aligned", np.array_equal(found, expected))
        primary = ref["primary"][mask].astype(bool)
        result["primary_n"] = int(primary.sum())
        if len(primary) == len(va):
            result["primary_labels"] = finite_stats(labels[va][primary])
        if prior.get("primary_rows") is not None:
            a.check(f"fold{fold}_primary_count", int(primary.sum()) == prior["primary_rows"])
    else:
        a.report["unresolved"].append(f"Fold {fold}: no reference path found.")
    return result


def run(a, args):
    native = a.npz("datasets/clear_ec_official/cache_native_meta_512.npz")
    full = a.npz("datasets/clear_ec_fullframe_20260904/fullframe_meta.npz")
    a.required(native, ("ids", "Y"), "native metadata")
    a.required(full, ("ids",), "full metadata")
    ids, labels = native["ids"].astype(str), native["Y"]
    n = len(ids)
    if ids.ndim != 1 or labels.shape != (n, 3):
        raise ValueError(f"Unexpected ID/label dimensions: {ids.shape}, {labels.shape}")
    a.check("unique_cache_ids", len(np.unique(ids)) == n)
    a.check("all_targets_finite", np.isfinite(labels).all())
    a.check("all_targets_nonnegative", (labels >= 0).all())
    a.check("full_native_id_order", np.array_equal(ids, full["ids"].astype(str)))
    if "Y" in full:
        a.check("full_native_labels_identical", np.array_equal(labels, full["Y"]))
    else:
        a.report["unresolved"].append("Full metadata has no Y; label equality cannot be checked.")
    a.report["dataset"] = dict(images=n, labels=finite_stats(labels),
                                label_order=list(METRICS), label_transform="none; raw audit")
    csv_path = a.identify("datasets/clear_ec_official/final_train_ids.csv")
    with csv_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = list(reader.fieldnames or []), list(reader)
    a.report["csv"] = dict(columns=fields, rows=len(rows))
    a.required(dict.fromkeys(fields), ("ID", *METRICS), "source label CSV")
    grouped = defaultdict(list)
    for row in rows:
        grouped[clean_id(row["ID"])].append(row)
    a.check("source_csv_normalized_ids_unique", all(len(v) == 1 for v in grouped.values()),
            duplicate_keys=sum(len(v) > 1 for v in grouped.values()))
    normalized = [clean_id(i) for i in ids]
    a.check("normalized_cache_ids_unique", len(set(normalized)) == n)
    missing = [key for key in normalized if key not in grouped]
    a.check("all_cache_ids_in_source_csv", not missing, missing_rows=len(missing))
    normalized_set = set(normalized)
    a.report["csv"]["rows_not_in_cache"] = sum(key not in normalized_set for key in grouped)
    slides = None
    if not missing and all(len(grouped[key]) == 1 for key in normalized):
        aligned = [grouped[key][0] for key in normalized]
        original = np.array([[float(row[m]) for m in METRICS] for row in aligned], dtype=labels.dtype)
        a.check("cache_labels_equal_source_csv", np.array_equal(original, labels),
                mismatched_entries=int(np.count_nonzero(original != labels)))
        if "slide_id" in fields and all(str(row["slide_id"]).strip() for row in aligned):
            slides = np.array([str(row["slide_id"]) for row in aligned])
            a.report["dataset"]["slide_groups"] = len(set(slides))
        else:
            a.report["unresolved"].append("Missing/empty slide_id mapping; slide leakage not audited.")
    a.report["grouping"] = dict(patient_identity="unresolved",
                                 reason="No verified patient-to-image mapping contract supplied; names/prefixes are not patient IDs.",
                                 near_duplicates="not exhaustively audited")
    a.report["unresolved"].append("Patient independence and near-duplicate leakage remain unverified.")
    manifest = a.json("datasets/clear_ec_fullframe_20260904/fullframe_manifest.json")
    a.report["full_cache_provenance"] = {k: manifest.get(k) for k in (
        "cache_shape", "archive_sha256", "central_pixels_exact_match", "archive_labels_match_cache")}
    geometry_path = a.path("datasets/clear_ec_fullframe_20260904/image_geometry.json")
    if geometry_path.exists():
        geometry = a.json(geometry_path)
        if isinstance(geometry, list) and all(isinstance(row, dict) for row in geometry):
            geometry_ids = [str(row.get("id", "")) for row in geometry]
            a.check("geometry_id_order", geometry_ids == ids.tolist())
            spacing_counts = defaultdict(int)
            for row in geometry:
                spacing_counts[json.dumps(row.get("spacing"), sort_keys=True)] += 1
            a.report["geometry"] = dict(rows=len(geometry),
                spacing_values=[dict(value=json.loads(k), rows=v) for k, v in sorted(spacing_counts.items())],
                physical_calibration_verified=False,
                caveat="MHA spacing may be a placeholder; no physical-unit/scale claim without acquisition provenance.")
        else:
            a.report["unresolved"].append("Unknown image_geometry schema; not interpreted.")
    fp_path = a.path("experiments/ssl_screen_20260911/preflight/pixel_fingerprints.npz")
    old_hashes = None
    if fp_path.exists():
        evidence = a.npz(fp_path)
        if "sha256" in evidence and evidence["sha256"].shape == (n,):
            old_hashes = evidence["sha256"].astype(str)
        else:
            a.report["unresolved"].append("Historical fingerprints have unknown keys/shape; not reused.")
    else:
        a.report["unresolved"].append("Historical pixel fingerprint evidence missing.")
    hashes = audit_pixels(a, n, manifest, old_hashes, args.skip_pixels)
    connected = group_components(slides, hashes) if slides is not None and hashes is not None else None
    if hashes is not None:
        buckets = defaultdict(list)
        for i, h in enumerate(hashes):
            buckets[h].append(i)
        repeated = [indices for indices in buckets.values() if len(indices) > 1]
        a.report["exact_duplicates"] = dict(groups=len(repeated), rows=sum(map(len, repeated)),
            extra_rows=sum(len(v) - 1 for v in repeated),
            conflicting_label_groups=sum(not np.all(labels[v] == labels[v[0]]) for v in repeated),
            evidence="current full-image SHA256" if not args.skip_pixels else "historical unverified positional SHA256")
    previous = a.json(args.split_manifest)
    a.required(previous, ("folds",), "split manifest")
    a.report["folds"] = {}
    for fold in ("2", "3"):
        try:
            a.report["folds"][fold] = audit_fold(a, fold, previous["folds"][fold], ids, labels, slides, hashes, connected)
        except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            a.report["errors"].append(f"fold{fold}: {type(exc).__name__}: {exc}")
    # Check hashes only for evidence this audit actually read; no checkpoint I/O.
    for relative, expected in previous.get("input_sha256", {}).items():
        path = str(a.path(relative))
        observed = a.report["inputs"].get(path, {}).get("sha256")
        if observed is not None:
            a.check(f"historical_input_sha:{relative}", observed == expected)
    candidates = []
    for directory in (a.root / "code", a.root / "evaluation", a.root / "evaluate"):
        if directory.exists():
            candidates.extend(str(p.relative_to(a.root)) for p in directory.rglob("*.py")
                              if "evaluat" in p.name.lower() or "metric" in p.name.lower())
    a.report["zero_target_policy"] = dict(status="unresolved", official_evaluator_verified=False,
        rationale="Filename matches and historical local scoring implementations do not establish official zero-target behavior.",
        local_candidate_paths=sorted(candidates),
        historical_policy="phase2_cpsam_scale_20261001.py excludes target<=0 separately for each metric; not adopted as official policy")
    if args.official_evaluator:
        p = a.identify(args.official_evaluator)
        lines = p.read_text().splitlines()
        a.report["zero_target_policy"]["provided_source"] = dict(path=str(p), sha256=sha(p),
            relevant_lines=[dict(line=i + 1, text=s[:300]) for i, s in enumerate(lines)
                            if any(term in s.lower() for term in ("mape", "percentage", "epsilon", "zero"))][:50],
            note="Manual provenance and behavior review still required; source is not executed.")
    a.report["unresolved"].append("Official zero-target metric policy is not verified; do not silently exclude or epsilon-replace targets.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, required=True, help="New JSON path; existing file is never replaced")
    parser.add_argument("--split-manifest", default="experiments/phase2_metric35_ssl_20260928/manifest.json")
    parser.add_argument("--official-evaluator", help="Optional local source; hash/excerpts only, never executed")
    parser.add_argument("--skip-pixels", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.out if args.out.is_absolute() else root / args.out
    if out.exists():
        parser.error(f"Refusing to overwrite existing output: {out}")
    # The caller creates the output directory; do not mutate historic trees.
    if not out.parent.is_dir():
        parser.error(f"Output directory must already exist: {out.parent}")
    a = Audit(root)
    try:
        run(a, args)
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        a.report["errors"].append(f"{type(exc).__name__}: {exc}")
    failed = [c["name"] for c in a.report["checks"] if not c["passed"]]
    a.report["failed_checks"] = failed
    a.report["status"] = "incomplete" if a.report["errors"] else "failed" if failed else "checks_passed_with_policy_gaps"
    a.report["accuracy_training_authorized_by_audit"] = False
    with out.open("x") as stream:
        json.dump(a.report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status=a.report["status"], output=str(out), failed_checks=failed,
                          error_count=len(a.report["errors"]), unresolved_count=len(a.report["unresolved"]))))
    return 3 if a.report["errors"] else 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
