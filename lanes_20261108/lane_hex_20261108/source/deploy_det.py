"""Full-data detectors for ridgeev_fall16 (HEX lane, 2026-11-10): runs green_track_20261028/source/green_deploy_train_20261030.py
(same recipe as its deploy/detectors/sg2_s46..50: full-data recipe A encoder, 1000 updates, batch 2) with the output folder moved
to this lane and, for the two sigma variants, the target sigma changed as their fold runs did (8 px -> 3.5 / 5 px).
usage: deploy_det.py NAME --seed S [--sigma X]  ->  lane_hex_20261108/deploy/detectors/NAME/sg2_s{S}.pt (+ .json)"""
import sys, argparse
from pathlib import Path
sys.path.insert(0, '/workspace/clear-ec-research/experiments/green_track_20261028/source')
import green_deploy_train_20261030 as DT
ap = argparse.ArgumentParser(); ap.add_argument('name'); ap.add_argument('--seed', type=int, required=True); ap.add_argument('--sigma', type=float, default=None); a = ap.parse_args()
DT.DEPLOY = Path('experiments/lane_hex_20261108/deploy') / 'detectors' / a.name
if a.sigma is not None: DT.spatial.SIGMA = a.sigma
print('deploy_det', a.name, 'seed', a.seed, 'sigma', DT.spatial.SIGMA, 'out', DT.DEPLOY / 'detectors', flush=True)
DT.main(argparse.Namespace(kind='sg2', seed=a.seed, encoder=DT.FULLDATA_ENCODER))
