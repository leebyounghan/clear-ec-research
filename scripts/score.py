#!/usr/bin/env python3
"""Score predictions.csv against a label CSV with columns case, cell_density, coefficient_of_variation, hexagonality.
Same rule as the organisers' evaluate_results: per-row absolute percentage error, rounded to one decimal, rows with a zero or
missing label skipped, mean per metric, and the score is the mean of the three. Usage: python3 scripts/score.py predictions.csv labels.csv"""
import csv, math, sys

pred = {r['case']: r for r in csv.DictReader(open(sys.argv[1]))}
lab = {r['case']: r for r in csv.DictReader(open(sys.argv[2]))}
keys = ('cell_density', 'coefficient_of_variation', 'hexagonality'); res = {}
for k in keys:
    e = []
    for c, r in lab.items():
        try: g = float(r[k])
        except ValueError: continue
        if c not in pred or not math.isfinite(g) or g == 0: continue
        e.append(round(abs(float(pred[c][k]) - g) / g * 100, 1))
    res[k] = sum(e) / len(e); print(f'{k}: MAPE {res[k]:.3f} over {len(e)} rows')
print(f'mean: {sum(res.values()) / 3:.4f}')
