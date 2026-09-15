"""Early checksum gate for a fully received model while other ranges download."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "submission_candidate_20260909"
PARTS = ROOT / "provenance/transfer_ranges"
parser = argparse.ArgumentParser()
parser.add_argument("name")
args = parser.parse_args()
plan = json.loads((PARTS / "plan.json").read_text())
item = plan[args.name]
target = ROOT / "model" / args.name
prefix = target.with_name(target.name + ".partial")
assert not target.exists()
assert prefix.stat().st_size == item["prefix"]
segments = []
for offset in range(item["prefix"], item["bytes"], 2 * 1024**2):
    count = min(2 * 1024**2, item["bytes"] - offset)
    path = PARTS / args.name / f"{offset}.part"
    if not path.exists() or path.stat().st_size != count:
        print("Not fully received yet", args.name)
        raise SystemExit(2)
    segments.append(path)
temporary = target.with_name(target.name + ".assembling")
h = hashlib.sha256()
with temporary.open("xb") as out:
    for path in [prefix] + segments:
        with path.open("rb") as source:
            for block in iter(lambda: source.read(8 * 1024**2), b""):
                out.write(block)
                h.update(block)
assert temporary.stat().st_size == item["bytes"] and h.hexdigest() == item["sha256"]
temporary.chmod(0o644)
temporary.rename(target)
backup = PARTS / (args.name + ".prefix.partial")
assert not backup.exists()
prefix.rename(backup)
print("Early full checkpoint SHA256 verified:", args.name, h.hexdigest())
