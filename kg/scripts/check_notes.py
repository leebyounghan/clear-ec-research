#!/usr/bin/env python3
"""Validate specific notes against the schema (safe to run while other sessions edit other notes).

Usage: /usr/bin/python3 scripts/check_notes.py <note.md> [more notes...]
Checks: YAML parses, id == filename, required fields, controlled values, known edge keys, link targets exist
(any note in wiki/), edge target types. Paper notes: unknown link targets are warnings, not failures.
Exit code 1 if any note fails.
"""
import pathlib
import sys

import kglib
from kglib import LINK, WIKI, guess_type, link_ids, parse_note


def known_ids():
    ids = {}
    for p in kglib.note_paths():
        meta, _ = parse_note(p)
        t = (meta or {}).get("type") if isinstance(meta, dict) else None
        ids[p.stem] = t or guess_type(p.stem)
    return ids


def main(files):
    schema = kglib.load_schema()
    edges, ntypes, values = schema["edges"], schema["node_types"], schema.get("values", {})
    ids = known_ids()
    failed = 0
    for f in files:
        path = pathlib.Path(f)
        meta, body = parse_note(path)
        errs, warns = [], []
        if meta is None:
            errs.append("no frontmatter")
        elif "_yaml_error" in meta:
            errs.append(f"YAML error: {meta['_yaml_error']}")
        else:
            t = meta.get("type")
            is_paper = t == "paper"
            if meta.get("id") != path.stem:
                errs.append(f"id {meta.get('id')!r} != filename {path.stem!r}")
            if t not in ntypes:
                errs.append(f"unknown type {t!r}")
            for req in ntypes.get(t, {}).get("required", []):
                if meta.get(req) in (None, "", []):
                    (warns if is_paper else errs).append(f"missing required field: {req}")
            for key in ("status", "lane"):
                allowed = values.get(f"{t}.{key}")
                if allowed and meta.get(key) not in (None, "") and meta.get(key) not in allowed:
                    errs.append(f"{key}={meta.get(key)!r} not in {allowed}")
            if t in ("idea", "constraint") and meta.get("origin") not in values.get("origin", []):
                errs.append(f"origin must be one of {values.get('origin')}")
            if t == "experiment":
                for k in ("baseline", "folds"):
                    if str(meta.get(k) or "").strip().lower() in ("", "?", "unknown", "n/a"):
                        errs.append(f"experiment needs a real `{k}` (write 'not stated in source' and set ambiguous: true if so)")
            for key, val in meta.items():
                if key in edges:
                    allowed = edges[key].get("to")
                    for tgt, _ in link_ids(val, strict=is_paper):
                        if tgt not in ids:
                            (warns if is_paper else errs).append(f"unknown link target in {key}: {tgt}")
                        elif allowed and ids[tgt] not in allowed:
                            errs.append(f"{key} -> {tgt} has type {ids[tgt]}, allowed {allowed}")
            for tgt in set(LINK.findall(body)):
                if tgt.strip() not in ids:
                    (warns if is_paper else errs).append(f"unknown body link: [[{tgt.strip()}]]")
        if errs:
            failed += 1
            print(f"FAIL {f}")
        else:
            print(f"ok   {f}" + (f"  ({len(warns)} warnings)" if warns else ""))
        for x in errs:
            print(f"   - {x}")
        for x in warns:
            print(f"   ~ {x}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
