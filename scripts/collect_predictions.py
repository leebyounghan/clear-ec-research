#!/usr/bin/env python3
"""Collect <output_dir>/<case>/{cell-density,coefficient-of-variation,hexagonality}.json into <output_dir>/predictions.csv.
The mode column is 'recipe' for a normal prediction and 'constant' when the fallback values were written."""
import json, sys
from pathlib import Path

out = Path(sys.argv[1]); rows = ['case,cell_density,coefficient_of_variation,hexagonality,mode,seconds']
for d in sorted(p for p in out.iterdir() if (p / 'cell-density.json').is_file()):
    v = [json.loads((d / f'{k}.json').read_text()) for k in ('cell-density', 'coefficient-of-variation', 'hexagonality')]
    mode, sec = '', ''
    try:
        log = json.loads((d / 'run_log.json').read_text().strip().splitlines()[-1]); mode, sec = log.get('mode', ''), log.get('seconds', '')
    except Exception:
        pass
    rows.append(f'{d.name},{v[0]},{v[1]},{v[2]},{mode},{sec}')
(out / 'predictions.csv').write_text('\n'.join(rows) + '\n')
print(f'{len(rows) - 1} cases -> {out / "predictions.csv"}')
