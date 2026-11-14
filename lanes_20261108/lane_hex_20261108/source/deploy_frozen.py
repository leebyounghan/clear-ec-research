"""Full-data frozen heads for ridgeev_fall16 (HEX lane, 2026-11-10): green_track_20261028/source/green_frozen_heads_20261031.py
with the frozen encoder set to the full-data recipe A s46 encoder (as its deploy/fallback_frozen/README.md prescribes) and the
output moved to this lane. The 'fold0' folder name is kept by the trainer; the head is the full-data one.
usage: deploy_frozen.py --kind {sg2f,sg2pref} --seed S --updates U  ->  lane_hex_20261108/deploy/frozen_heads/<kind>_s<S>/fold0/head.pt"""
import sys, argparse
from pathlib import Path
sys.path.insert(0, '/workspace/clear-ec-research/experiments/green_track_20261028/source')
import green_frozen_heads_20261031 as FH
FULL = 'experiments/fulldata_recipeA_20261028/regression_std_s46/runs/fold99/cpdino_vitb/epoch_20.pt'
ap = argparse.ArgumentParser(); ap.add_argument('--kind', required=True); ap.add_argument('--seed', type=int, required=True); ap.add_argument('--updates', type=int, required=True); a = ap.parse_args()
FH.ST.ENCODERS[0] = FULL; FH.ROOT = Path('experiments/lane_hex_20261108/deploy/frozen_heads')
print('deploy_frozen', a.kind, a.seed, a.updates, 'encoder', FULL, flush=True)
FH.main(argparse.Namespace(command='train', fold=0, kind=a.kind, seed=a.seed, updates=a.updates))
