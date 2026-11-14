#!/usr/bin/env python3
"""List lane RESULTS.tsv rows that no experiment node cites yet (for incremental ingest).

Usage:
  perl -e 'alarm 90; exec @ARGV' kubectl exec clear-ec-research-4gpu-1104 -- \
      bash -c 'cat /workspace/clear-ec-research/experiments/lane_cv_20261108/RESULTS.tsv' > /tmp/cv.tsv
  /usr/bin/python3 scripts/unrecorded_rows.py lane_cv_20261108 /tmp/cv.tsv [--uncited] [--all]

RESULTS.tsv files are append-only, so line numbers are stable. By default it prints rows after `reviewed_through`
in sources.yaml (new since the last ingest). --uncited prints every row that no note's `source` names
("<lane folder>/RESULTS.tsv line N"). Repeated HEX 'chain step' re-runs are skipped unless --all.
After ingesting, raise reviewed_through in sources.yaml.
"""
import re
import sys

import kglib
from kglib import parse_note


def main(argv):
    if len(argv) < 2:
        raise SystemExit(__doc__)
    lane, path = argv[0], argv[1]
    show_all = "--all" in argv
    uncited_mode = "--uncited" in argv
    import yaml
    src = yaml.safe_load(open(kglib.ROOT / "sources.yaml", encoding="utf-8")) or {}
    through = int(((src.get("lanes") or {}).get(lane) or {}).get("reviewed_through") or 1)
    cited = set()
    pat = re.compile(re.escape(lane) + r"/RESULTS\.tsv line (\d+)")
    for p in kglib.note_paths():
        meta, _ = parse_note(p)
        if isinstance(meta, dict):
            cited |= {int(n) for n in pat.findall(str(meta.get("source", "")))}
    lines = open(path, encoding="utf-8").read().splitlines()
    seen_steps = set()
    for i, ln in enumerate(lines[1:], start=2):
        if i in cited or (not uncited_mode and i <= through):
            continue
        cols = ln.split("\t")
        name = cols[1] if len(cols) > 1 else ln
        if not show_all and name.startswith("chain step"):
            key = (name, cols[3] if len(cols) > 3 else "")
            if key in seen_steps:
                continue
            seen_steps.add(key)
        print(f"line {i}: {kglib.short(ln.replace(chr(9), ' | '), 220)}")
    print(f"\n{len(cited)} rows of {lane} cited by notes; {len(lines)} lines in the file; reviewed through line {through}"
          + (" (showing all uncited rows)" if uncited_mode else " (showing new rows only)"))


if __name__ == "__main__":
    main(sys.argv[1:])
