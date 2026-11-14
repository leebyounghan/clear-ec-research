"""Verifier L5 check for D-CV-001 (read-only use of the CV lane's code): run the CV lane's strict search (cv_search_20261111.py, fixed6 component
set, same coordinate descent and 0.125 grid) with no outer fold, i.e. its score_all objective (inner fold g stretch on folds != g), and compare the
resulting weights with deploy/cv_deploy_params.json; also score both weight sets with the same objective."""
import sys, json, numpy as np
sys.argv = ['cv_search_20261111.py', 'pool_20261111.npz']
sys.path.insert(0, '/workspace/clear-ec-research/experiments/lane_cv_20261108/source')
import cv_search_20261111 as S
six = [S.NAMES.index(n) for n in S.NAMES if n in ('dino', 'fz:spectrum', 'fz:cov_sqrt', 'fz:autocorr') or n.startswith('graph:tta_s46_tta4_5f+') or n.startswith('graph:gord_s46_tta4_5f+')]
w, sc = S.coord(np.zeros(S.K), S.score_all, six)
dep = json.load(open('/workspace/clear-ec-research/experiments/lane_cv_20261108/deploy/cv_deploy_params.json'))['weights']
short = {'dino': 'dino', 'fz:spectrum': 'fz:spectrum', 'fz:cov_sqrt': 'fz:cov_sqrt', 'fz:autocorr': 'fz:autocorr'}
wd = np.zeros(S.K)
for k in six:
    n = S.NAMES[k]; key = short.get(n, 'graph:plain' if n.startswith('graph:tta_') else 'graph:order')
    wd[k] = dep[key]
res = dict(strict_no_outer_fold={S.NAMES[k][:40]: float(w[k]) for k in six}, deploy={S.NAMES[k][:40]: float(wd[k]) for k in six},
           score_all_strict_weights=round(float(sc), 5), score_all_deploy_weights=round(float(S.score_all(wd)), 5), equal=bool(np.allclose(w[six], wd[six])))
print(json.dumps(res, indent=1))
json.dump(res, open('/workspace/clear-ec-research/experiments/lane_cd_20261108/reads/verify_cv_l5_20261111.json', 'w'), indent=1)
