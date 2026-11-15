#!/usr/bin/env python3
"""Check checkpoints/ against configs/checkpoint_sha256.txt (sha256, size in bytes, relative path; 443 files, 16,249,746,790 bytes).
Usage: python3 scripts/verify_checkpoints.py [checkpoint_dir] [--quick]   (--quick checks presence and size only)"""
import hashlib, sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
args = [a for a in sys.argv[1:] if not a.startswith('--')]
ckpt = Path(args[0]) if args else root / 'checkpoints'
quick = '--quick' in sys.argv
bad = 0
rows = [l.split(None, 2) for l in (root / 'configs' / 'checkpoint_sha256.txt').read_text().splitlines() if l.strip()]
for i, (sha, size, rel) in enumerate(rows, 1):
    p = ckpt / rel
    if not p.is_file():
        print(f'MISSING {rel}'); bad += 1; continue
    if p.stat().st_size != int(size):
        print(f'SIZE    {rel}: {p.stat().st_size} != {size}'); bad += 1; continue
    if not quick:
        h = hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda: f.read(1 << 24), b''): h.update(b)
        if h.hexdigest() != sha:
            print(f'SHA256  {rel}'); bad += 1
    if i % 50 == 0: print(f'  {i}/{len(rows)} checked', flush=True)
print(f'{len(rows) - bad} of {len(rows)} files OK' + (' (presence and size only)' if quick else ''))
sys.exit(1 if bad else 0)
