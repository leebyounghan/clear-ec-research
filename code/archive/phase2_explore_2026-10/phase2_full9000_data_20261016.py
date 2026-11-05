#!/usr/bin/env python3
"""Preparation-only full-native 9000 image reader; no model or training policy.

Full9000Reader reads NPZ ``ids`` only, never Y/Y_train or CSV label columns.
load_original_targets is a separate, explicit regression-side operation. It
returns unsmoothed Y and masks, including the two source HEX zeros, unchanged.

Preflight normally reuses the pinned historical 8954 pixel audit and freshly
checks the 46 extra MHA files. Raw size and an old audit DO NOT establish current
8954 content integrity. A future training launch must call
verify_current_raw_sha(workspace) immediately before using these inputs (or
provide equally strong independently reviewed immutability evidence). Passing
this preparation audit does not authorize or select a full-data fit or a zero
label policy. No validation score exists for an all-data fit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import operator
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


NATIVE = (972, 1296)
OLD_N, FINAL_N, EXTRA_N = 8954, 9000, 46
RAW_ROOT = "datasets/clear_ec_fullframe_20260904"
FINAL_ROOT = "datasets/clear_ec_final9000_20260904"
RAW_PATH = RAW_ROOT + "/img_full_raw.dat"
CENTRAL_PATH = FINAL_ROOT + "/img_native_512.dat"
ARCHIVE_SHA = "799a833e53f33196882d5e5b014f6309ec9c7d5250110093dda88a7dfb2fe263"
RAW_SHA = "93e67ac6df5887a1a5396c85e236b8398297dd3e728cdb6513238c1628c1c88a"
OLD_AUDIT_SHA = "98c04ceb0c01f852995be4089709ecf0e6c453e4e238830f0f31ec4dbc9ec07e"
ZERO_IDS = {"2019-0064-ODCN", "2019-0064-OSCN"}
INPUTS = {
    "old_meta": ("datasets/clear_ec_official/cache_native_meta_512.npz",
                 "956f448a2de4d65b4e091d78f56bdde09aab9554e3ab027fc7fe58f8b69625e6"),
    "raw_meta": (RAW_ROOT + "/fullframe_meta.npz",
                 "fceab01753e03dcfbc871712344fbd403b5b945760f9c2f78abb4968975defb9"),
    "raw_manifest": (RAW_ROOT + "/fullframe_manifest.json",
                     "1c7b7e3d9a5e8fc9565ce796e5f554a0049cb05664567b4501a8e7ab8d9937b4"),
    "source_csv": (RAW_ROOT + "/source_metadata/final_train_ids.csv",
                   "809c8b6c657abaa1812daed30c88125a1868ccf65684f88c53d7e1e30b29fc1e"),
    "reconciled_csv": (RAW_ROOT + "/source_metadata/final_train_ids_9000_reconciled.csv",
                       "9cb97a7b4c25ef86d65d0c95f248bc6f8f05ef554309fbd22e586afde4d1f09b"),
    "alias_audit": (RAW_ROOT + "/raw_id_alias_audit.json",
                    "a42449ed563e046a07a8ed00736f604da553838b0ac516a11093bedf2bd56d0e"),
    "final_manifest": (FINAL_ROOT + "/manifest.json",
                       "20b8ee12055786dcdcb23fbcbb6fa033ad6ba9b56a393b568e51f98b9136695f"),
    "final_meta": (FINAL_ROOT + "/meta.npz",
                   "c19508db0cdeaf3fb28c5f99d669eeac47639bde4f7b04a2f117a522d7102bb4"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _stamp(path):
    s = Path(path).stat()
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


def sha256(path):
    path = Path(path)
    before = _stamp(path)
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    require(_stamp(path) == before, f"file changed while hashing: {path}")
    return h.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text())


def _json_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def _pixel_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _verify_inputs(workspace):
    # Hashing NPZ bytes does not deserialize or access any target array.
    out = {}
    for name, (relative, expected) in INPUTS.items():
        path = workspace / relative
        actual = sha256(path)
        require(actual == expected, f"input hash mismatch: {relative}")
        out[name] = dict(path=relative, sha256=actual, bytes=path.stat().st_size)
    return out


def _load_ids(path, count):
    with np.load(path, allow_pickle=False) as meta:
        ids = meta["ids"]  # Deliberately do not enumerate/load Y or Y_train.
    require(ids.shape == (count,) and ids.dtype.kind in "US", "invalid IDs")
    ids = ids.astype(str)
    require(len(set(ids.tolist())) == count, "duplicate IDs")
    require(all(x and "/" not in x and "\\" not in x and x not in {".", ".."}
                for x in ids), "unsafe or empty ID")
    ids.flags.writeable = False
    return ids


def _index(workspace):
    ids = _load_ids(workspace / INPUTS["final_meta"][0], FINAL_N)
    old_ids = _load_ids(workspace / INPUTS["old_meta"][0], OLD_N)
    raw_ids = _load_ids(workspace / INPUTS["raw_meta"][0], OLD_N)
    require(np.array_equal(old_ids, raw_ids), "old/raw ID order mismatch")
    lookup = {value: i for i, value in enumerate(old_ids)}
    mapping = np.asarray([lookup.get(value, -1) for value in ids], dtype=np.int64)
    require(int((mapping >= 0).sum()) == OLD_N, "old population not preserved")
    require(set(mapping[mapping >= 0].tolist()) == set(range(OLD_N)),
            "old mapping is not a bijection")
    extra_indices = np.flatnonzero(mapping < 0)
    extra_ids = set(ids[extra_indices].tolist())
    alias = _json(workspace / INPUTS["alias_audit"][0])
    raw_manifest = _json(workspace / INPUTS["raw_manifest"][0])
    final_manifest = _json(workspace / INPUTS["final_manifest"][0])
    require(len(extra_indices) == EXTRA_N, "extra count mismatch")
    require(alias["normalized_bijection"] is True and
            alias["raw_count"] == alias["label_count"] == FINAL_N,
            "alias inventory mismatch")
    require(len(alias["aliases"]) == EXTRA_N and
            {a["raw_id"] for a in alias["aliases"]} == extra_ids ==
            set(raw_manifest["extra_raw_ids"]), "alias extra-ID mismatch")
    require(alias["archive_sha256"] == raw_manifest["archive_sha256"] ==
            final_manifest["archive_sha256"] == ARCHIVE_SHA, "archive chain mismatch")
    require(raw_manifest["cache_shape"] == [OLD_N, *NATIVE] and
            raw_manifest["cache_dtype"] == "uint8" and
            raw_manifest["matched_cache_rows"] == OLD_N and
            raw_manifest["raw_file_count"] == FINAL_N, "native manifest mismatch")
    require(final_manifest["n"] == FINAL_N and
            final_manifest["matched_existing"] == OLD_N and
            final_manifest["source_csv_sha256"] == INPUTS["reconciled_csv"][1],
            "final manifest mismatch")
    require((workspace / RAW_PATH).stat().st_size == OLD_N * int(np.prod(NATIVE)),
            "raw byte size mismatch (this is not a content hash)")
    mapping.flags.writeable = False
    summary = dict(ordered_ids_sha256=_json_sha(ids.tolist()),
                   old_index_mapping_sha256=_pixel_sha(mapping.astype("<i8")),
                   count=FINAL_N, old_count=OLD_N, extra_count=EXTRA_N,
                   raw_shape=list(NATIVE), raw_dtype="uint8")
    return ids, mapping, summary


def _read_mha(path, expected_shape=NATIVE):
    # Reject external-data MHA before invoking the image library.
    header = {}
    with Path(path).open("rb") as stream:
        while stream.tell() < 65536:
            line = stream.readline(65537)
            require(line and len(line) <= 65536, "missing/oversized MHA header")
            text = line.decode("ascii").strip()
            if not text or text.startswith("#"):
                continue
            key, sep, value = text.partition("=")
            require(bool(sep), "malformed MHA header")
            key, value = key.strip(), value.strip()
            require(key not in header, "duplicate MHA header field")
            header[key] = value
            if key == "ElementDataFile":
                break
    require(header.get("ElementDataFile") == "LOCAL", "MHA must be self-contained")
    require(header.get("NDims") == "2" and header.get("ElementType") == "MET_UCHAR"
            and header.get("ElementNumberOfChannels", "1") == "1",
            "MHA must be scalar two-dimensional uint8")
    require(tuple(map(int, header["DimSize"].split())) == tuple(reversed(expected_shape)),
            "MHA native shape mismatch")
    import SimpleITK as sitk
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    image = sitk.ReadImage(str(path))
    require(image.GetDimension() == 2 and image.GetNumberOfComponentsPerPixel() == 1,
            "MHA frame/channel count mismatch")
    pixels = sitk.GetArrayFromImage(image)
    require(pixels.shape == expected_shape and pixels.dtype == np.uint8,
            "MHA native array mismatch")
    geometry = dict(spacing=list(image.GetSpacing()), origin=list(image.GetOrigin()),
                    direction=list(image.GetDirection()))
    require(all(np.isfinite(v).all() for v in geometry.values()) and
            all(v > 0 for v in geometry["spacing"]), "invalid MHA geometry")
    # Preserve stored pixel order; do not reorient or resample.
    return pixels, dict(header=header, geometry=geometry, frames=1,
                        orientation_operation="none")


def _crop(pixels, size=512):
    require(pixels.shape[0] >= size and pixels.shape[1] >= size, "crop too large")
    r, c = ((x - size) // 2 for x in pixels.shape)
    return pixels[r:r + size, c:c + size]


def _original_target_values(path, count):
    with np.load(path, allow_pickle=False) as meta:
        targets = meta["Y"]  # Never Y_train (historical slide averages).
    require(targets.shape == (count, 3) and targets.dtype == np.float32 and
            np.isfinite(targets).all() and (targets >= 0).all(), "invalid original Y")
    return targets


def _target_summary(ids, targets):
    positive = targets > 0
    return dict(label_order=["CD", "CV", "HEX"], target_transform="none",
                targets_sha256=_pixel_sha(targets), positive_rows=int(positive.all(1).sum()),
                positive_per_target=positive.sum(0).tolist(),
                zero_per_target=(targets == 0).sum(0).tolist(),
                nonpositive_ids=ids[~positive.all(1)].tolist(),
                zero_policy="not selected; values preserved, no filtering or epsilon")


def _audit_labels(workspace, ids, mapping):
    def rows(name):
        with (workspace / INPUTS[name][0]).open(newline="") as stream:
            return list(csv.DictReader(stream))
    source, corrected = rows("source_csv"), rows("reconciled_csv")
    require(len(source) == len(corrected) == FINAL_N, "CSV row count mismatch")
    require([r["ID"] for r in corrected] == ids.tolist(), "CSV/final metadata order mismatch")
    require(len({r["ID"] for r in source}) == FINAL_N, "source IDs not unique")
    raw_ids = {p.stem for p in (workspace / RAW_ROOT / "train_mha").glob("*.mha")}
    require(raw_ids == set(ids.tolist()), "9000 MHA inventory mismatch")
    source_ids = {r["ID"] for r in source}
    exact = raw_ids & source_ids
    canonical = lambda v: re.sub(r"[\s-]", "", v.removesuffix(".mha").removesuffix(".bmp"))
    unmatched = {canonical(value): value for value in raw_ids - exact}
    source_unmatched = {canonical(value): value for value in source_ids - exact}
    require(len(unmatched) == len(source_unmatched) == EXTRA_N and
            set(unmatched) == set(source_unmatched) and
            not set(unmatched).intersection(canonical(i) for i in exact),
            "alias rule is not a unique collision-free bijection")
    aliases = []
    for original, fixed in zip(source, corrected):
        raw_id = original["ID"] if original["ID"] in exact else unmatched[canonical(original["ID"])]
        require(fixed == dict(original, ID=raw_id, source_csv_id=original["ID"]),
                "reconciliation changed a source field other than ID")
        if raw_id != original["ID"]:
            aliases.append(dict(raw_id=raw_id, source_csv_id=original["ID"],
                                slide_id=original["slide_id"]))
    require(aliases == _json(workspace / INPUTS["alias_audit"][0])["aliases"],
            "alias audit does not reproduce exact-first mapping")
    targets = _original_target_values(workspace / INPUTS["final_meta"][0], FINAL_N)
    csv_targets = np.asarray([[float(r[k]) for k in ("CD", "CV", "HEX")]
                              for r in corrected], dtype=np.float32)
    require(np.array_equal(targets, csv_targets), "original CSV/final Y mismatch")
    old_targets = _original_target_values(workspace / INPUTS["old_meta"][0], OLD_N)
    raw_targets = _original_target_values(workspace / INPUTS["raw_meta"][0], OLD_N)
    shared = mapping >= 0
    require(np.array_equal(old_targets, raw_targets) and
            np.array_equal(targets[shared], old_targets[mapping[shared]]),
            "shared original targets differ")
    summary = _target_summary(ids, targets)
    require(summary["positive_rows"] == 8998 and summary["zero_per_target"] == [0, 0, 2]
            and set(summary["nonpositive_ids"]) == ZERO_IDS, "unexpected source zero inventory")
    return targets, corrected, summary


def verify_current_raw_sha(workspace):
    """Expensive explicit launch gate: hash current 8954 native bytes, no labels.

    Call immediately before any new training; an old PASS is not a perpetual
    immutability guarantee. No outputs are written and no fit is launched.
    """
    path = Path(workspace).resolve() / RAW_PATH
    require(path.stat().st_size == OLD_N * int(np.prod(NATIVE)), "raw size mismatch")
    actual = sha256(path)
    require(actual == RAW_SHA, "CURRENT full-native raw SHA mismatch")
    return dict(path=RAW_PATH, sha256=actual, bytes=path.stat().st_size,
                checked_utc=_utc(), current_raw_sha256_verified=True)


def preflight(workspace, existing_audit, output, *, verify_raw=False):
    workspace, output = Path(workspace).resolve(), Path(output)
    require(not output.exists(), "audit output already exists; overwriting is forbidden")
    inputs = _verify_inputs(workspace)
    existing_audit = Path(existing_audit)
    require(sha256(existing_audit) == OLD_AUDIT_SHA, "historical pixel audit hash mismatch")
    old = _json(existing_audit)
    require(old["status"] == "shared_checks_passed" and not old["failed_shared_checks"] and
            old["pixels"]["verified"] is True and old["pixels"]["n"] == OLD_N and
            old["pixels"]["mismatch_rows"] == 0 and
            old["pixels"]["whole_cache_sha256"] == RAW_SHA,
            "historical 8954 pixel evidence invalid")
    for key in ("old_meta", "raw_meta", "raw_manifest"):
        relative, expected = INPUTS[key]
        require(old["inputs"][relative]["sha256"] == expected, "old audit input binding mismatch")
    ids, mapping, index = _index(workspace)
    targets, rows, labels = _audit_labels(workspace, ids, mapping)
    central_path = workspace / CENTRAL_PATH
    require(central_path.stat().st_size == FINAL_N * 512 * 512,
            "final9000 central512 byte size mismatch")
    central = np.memmap(central_path, dtype=np.uint8, mode="r", shape=(FINAL_N, 512, 512))
    raw = np.memmap(workspace / RAW_PATH, dtype=np.uint8, mode="r", shape=(OLD_N, *NATIVE))
    extras = []
    for i in np.flatnonzero(mapping < 0).tolist():
        relative = RAW_ROOT + "/train_mha/" + str(ids[i]) + ".mha"
        path = workspace / relative
        before = _stamp(path)
        pixels, metadata = _read_mha(path)
        digest = sha256(path)
        require(_stamp(path) == before, "MHA changed during preflight")
        require(np.array_equal(_crop(pixels), central[i]), f"extra central512 mismatch: {ids[i]}")
        extras.append(dict(index=i, id=str(ids[i]), path=relative,
                           bytes=path.stat().st_size, sha256=digest,
                           pixel_sha256=_pixel_sha(pixels), metadata=metadata,
                           central512_sha256=_pixel_sha(central[i]), central512_equal=True,
                           source_csv_id=rows[i]["source_csv_id"], original_label_matches_source_csv=True,
                           original_target_row_sha256=_pixel_sha(targets[i])))
    # Eight deterministic shared rows check the final index mapping only; they
    # do not replace whole-cache SHA or establish current integrity of all8954.
    shared_indices = np.flatnonzero(mapping >= 0)
    selected = shared_indices[np.linspace(0, OLD_N - 1, 8, dtype=np.int64)]
    shared = []
    for i in selected.tolist():
        require(np.array_equal(_crop(raw[mapping[i]]), central[i]), "shared central512 mismatch")
        shared.append(dict(index=i, id=str(ids[i]), old_index=int(mapping[i]),
                           central512_sha256=_pixel_sha(central[i]), central512_equal=True))
    del central, raw
    current = verify_current_raw_sha(workspace) if verify_raw else dict(
        path=RAW_PATH, expected_sha256=RAW_SHA, current_raw_sha256_verified=False,
        bytes=(workspace / RAW_PATH).stat().st_size,
        limitation="Only historical 8954 pixel audit reused; current whole raw not rehashed")
    report = dict(schema="clear_ec_full9000_data_v1", status="preparation_checks_passed",
                  created_utc=_utc(), source_sha256=sha256(__file__), inputs=inputs,
                  index=index, labels=labels, extras=extras, shared_central_checks=shared,
                  image_inventory_sha256=_json_sha(sorted(ids.tolist())),
                  historical_8954_pixel_evidence=dict(path=str(existing_audit),
                      sha256=OLD_AUDIT_SHA, checked_utc=old["created_utc"],
                      whole_cache_sha256=RAW_SHA, rows=OLD_N, mismatch_rows=0),
                  current_raw=current, current_raw_sha256_verified=bool(verify_raw),
                  central512_scope="46 extras plus 8 shared index checks; never model input; no full central-cache hash",
                  launch_gate_passed=False, training_selected=False, zero_policy_selected=False,
                  launch_requirement="Explicit immediate prelaunch verify_current_raw_sha PASS (or reviewed equivalent); preparation alone is insufficient",
                  archive_rehashed=False, historical_8954_rechecked_this_run=bool(verify_raw))
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return dict(status=report["status"], output=str(output), sha256=sha256(output),
                extra_images=EXTRA_N, current_raw_sha256_verified=bool(verify_raw),
                launch_gate_passed=False, training_started=False)


def _bound_index(workspace, audit_path, expected_audit_sha256):
    require(isinstance(expected_audit_sha256, str) and len(expected_audit_sha256) == 64,
            "expected audit SHA is required")
    require(sha256(audit_path) == expected_audit_sha256, "full9000 audit hash mismatch")
    report = _json(audit_path)
    require(report["schema"] == "clear_ec_full9000_data_v1" and
            report["status"] == "preparation_checks_passed" and
            report["source_sha256"] == sha256(__file__), "audit/source contract mismatch")
    require(_verify_inputs(workspace) == report["inputs"], "current input binding mismatch")
    ids, mapping, index = _index(workspace)
    require(index == report["index"], "current index binding mismatch")
    require(report["historical_8954_pixel_evidence"]["sha256"] == OLD_AUDIT_SHA and
            report["historical_8954_pixel_evidence"]["whole_cache_sha256"] == RAW_SHA,
            "pixel evidence binding mismatch")
    require([r["index"] for r in report["extras"]] == np.flatnonzero(mapping < 0).tolist(),
            "extra index binding mismatch")
    for record in report["extras"]:
        i = record["index"]
        require(record["id"] == ids[i] and record["path"] ==
                RAW_ROOT + "/train_mha/" + str(ids[i]) + ".mha", "extra path/ID mismatch")
    return report, ids, mapping


class _ImageRows:
    """Only pixel storage; private fixture seam does not bypass production pins."""
    def _open_pixels(self, raw_path, ids, mapping, extra_records, workspace, native_shape, old_n):
        self.ids, self.mapping = ids, mapping
        self._raw = np.memmap(raw_path, dtype=np.uint8, mode="r", shape=(old_n, *native_shape))
        self._extras = {}
        for record in extra_records:
            path = workspace / record["path"]
            before = _stamp(path)
            require(sha256(path) == record["sha256"], "current extra MHA hash mismatch")
            pixels, metadata = _read_mha(path, native_shape)
            require(_stamp(path) == before and _pixel_sha(pixels) == record["pixel_sha256"]
                    and metadata == record["metadata"], "extra pixels/header changed")
            pixels.flags.writeable = False
            self._extras[record["index"]] = pixels

    def __len__(self):
        return len(self.ids)

    def raw_frame(self, index):
        require(not isinstance(index, (bool, np.bool_)), "boolean image index")
        i = operator.index(index)
        require(0 <= i < len(self), "image index out of bounds")
        return self._raw[int(self.mapping[i])] if self.mapping[i] >= 0 else self._extras[i]


class Full9000Reader(_ImageRows):
    """Label-free native reader in final9000 metadata ID order.

    This reads the bound audit but never reads label arrays or CSV contents.
    It verifies all 46 extra files again. The 8954 memmap is size checked;
    call verify_current_raw_sha separately for an actual new-training gate.
    """
    def __init__(self, workspace, audit_path, *, expected_audit_sha256):
        workspace = Path(workspace).resolve()
        report, ids, mapping = _bound_index(workspace, audit_path, expected_audit_sha256)
        self._open_pixels(workspace / RAW_PATH, ids, mapping, report["extras"],
                          workspace, NATIVE, OLD_N)
        self.target_arrays_loaded = False
        self.provenance = dict(audit_sha256=expected_audit_sha256, index=report["index"],
            source_sha256=report["source_sha256"], target_arrays_loaded=False,
            current_raw_sha256_verified_by_this_reader=False,
            raw_integrity="historical audit only; separate immediate prelaunch hash gate required")


def load_original_targets(workspace, audit_path, *, expected_audit_sha256):
    """Explicit original-Y access. Never call this from label-free SSL training."""
    workspace = Path(workspace).resolve()
    report, ids, _ = _bound_index(workspace, audit_path, expected_audit_sha256)
    targets = _original_target_values(workspace / INPUTS["final_meta"][0], FINAL_N)
    counts = _target_summary(ids, targets)
    require(counts == report["labels"], "target audit mismatch")
    positive = targets > 0
    targets.flags.writeable = False
    return dict(ids=ids, targets=targets, positive_target_mask=positive,
                positive_row_mask=positive.all(1), counts=counts,
                provenance=dict(audit_sha256=expected_audit_sha256,
                    metadata_sha256=INPUTS["final_meta"][1], target_array="Y",
                    Y_train_loaded=False, target_transform="none", zero_policy_selected=False))


def synthetic_reader_tests():
    """Small CPU fixture only. Production data/hash gates cannot be overridden."""
    with tempfile.TemporaryDirectory(prefix="full9000-reader-") as directory:
        root = Path(directory)
        shape = (12, 16)
        old = np.arange(3 * np.prod(shape), dtype=np.uint16).reshape(3, *shape).astype(np.uint8)
        old.tofile(root / "raw.dat")
        ids = np.asarray(["extra-b", "old-c", "old-a", "extra-a", "old-b"])
        mapping = np.asarray([-1, 2, 0, -1, 1], dtype=np.int64)
        targets = np.asarray([[10, .1, .2], [20, .2, 0], [30, .3, .4],
                              [40, .4, .5], [50, .5, .6]], dtype=np.float32)
        # Accessing either sentinel array with allow_pickle=False fails. IDs and
        # explicit original-Y loading must work without touching these entries.
        np.savez(root / "ids_only.npz", ids=ids, Y=np.asarray([object()]), Y_train=np.asarray([object()]))
        np.savez(root / "targets.npz", ids=ids, Y=targets, Y_train=np.asarray([object()]))
        np.testing.assert_array_equal(_load_ids(root / "ids_only.npz", 5), ids)
        np.testing.assert_array_equal(_original_target_values(root / "targets.npz", 5), targets)
        records, expected = [], {1: old[2], 2: old[0], 4: old[1]}
        for i in (0, 3):
            pixels = (np.arange(np.prod(shape)).reshape(shape) + 19 * (i + 1)).astype(np.uint8)
            path = root / (str(ids[i]) + ".mha")
            header = ("ObjectType = Image\nNDims = 2\nBinaryData = True\n"
                      "BinaryDataByteOrderMSB = False\nCompressedData = False\n"
                      "TransformMatrix = 1 0 0 1\nOffset = 0 0\nElementSpacing = 1 1\n"
                      "DimSize = 16 12\nElementType = MET_UCHAR\nElementDataFile = LOCAL\n")
            path.write_bytes(header.encode() + pixels.tobytes())
            decoded, metadata = _read_mha(path, shape)
            np.testing.assert_array_equal(decoded, pixels)
            records.append(dict(index=i, path=path.name, sha256=sha256(path),
                                pixel_sha256=_pixel_sha(pixels), metadata=metadata))
            expected[i] = pixels
        reader = _ImageRows()
        reader._open_pixels(root / "raw.dat", ids, mapping, records, root, shape, 3)
        require(len(reader) == 5, "fixture row count")
        for i in range(5):
            np.testing.assert_array_equal(reader.raw_frame(i), expected[i])
            np.testing.assert_array_equal(_crop(reader.raw_frame(i), 8), expected[i][2:10, 4:12])
        for invalid in (-1, 5, True):
            try:
                reader.raw_frame(invalid)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid index accepted")
        summary = _target_summary(ids, targets)
        require(summary["positive_rows"] == 4 and summary["zero_per_target"] == [0, 0, 1]
                and targets[1, 2] == 0, "zero preservation fixture")
        for replacement in ("ElementDataFile = external.raw", "ElementType = MET_USHORT",
                            "NDims = 3", "DimSize = 15 12"):
            key = replacement.split(" = ")[0]
            bad_header = "\n".join(replacement if line.startswith(key + " = ") else line
                                    for line in header.rstrip().split("\n")) + "\n"
            bad = root / "bad.mha"
            bad.write_bytes(bad_header.encode() + pixels.tobytes())
            try:
                _read_mha(bad, shape)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid MHA accepted")
        # A one-byte mutation must be caught before reading an extra image.
        changed = root / records[0]["path"]
        data = changed.read_bytes()
        changed.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
        try:
            _ImageRows()._open_pixels(root / "raw.dat", ids, mapping, records, root, shape, 3)
        except ValueError:
            pass
        else:
            raise AssertionError("changed extra MHA accepted")
        del reader
    return dict(status="synthetic_reader_tests_passed", model_or_gpu_used=False,
                production_data_used=False, label_free_id_sentinel_passed=True,
                original_Y_and_zero_preserved=True, mapping_rows_checked=5,
                bad_MHA_cases_rejected=4, current_extra_mutation_rejected=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("preflight")
    prepare.add_argument("--workspace", required=True, type=Path)
    prepare.add_argument("--existing-audit", required=True, type=Path)
    prepare.add_argument("--output", required=True, type=Path)
    prepare.add_argument("--verify-current-raw-sha", action="store_true")
    verify = sub.add_parser("verify-current-raw")
    verify.add_argument("--workspace", required=True, type=Path)
    sub.add_parser("reader-tests")
    args = parser.parse_args()
    if args.command == "preflight":
        result = preflight(args.workspace, args.existing_audit, args.output,
                           verify_raw=args.verify_current_raw_sha)
    elif args.command == "verify-current-raw":
        result = verify_current_raw_sha(args.workspace)
    else:
        result = synthetic_reader_tests()
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
