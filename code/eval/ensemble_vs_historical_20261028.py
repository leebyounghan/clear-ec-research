"""Pre-registered decision input: recipe A seed ensemble vs the historical calibrated six on fold4 matched rows.

Reference (reproduced exactly on 2026-10-28): six = (5*current5 + ssl_sixth)/6 from
target_fold4_20260912/preflight/reference.npz and raw_ssl_next_20260913/ssl_runs/fold4/standard/oof.npz,
times the fixed Phase 1 scalars -> 8.8881 on the 1778 primary rows. Candidate: equal-weight mean of the
decoded fold4 recipe A members (seeds available), four-flip TTA if the TTA gate passed, scalar calibration
fitted on the seed-46 OOF of folds other than fold4 if the calibration gate passed. Reports the point rule
and a paired cluster bootstrap over id-prefix groups. CPU only; never promotes anything.
"""
import argparse
import datetime
import json
from pathlib import Path

import numpy as np

E = Path('experiments')
REF = E/'target_fold4_20260912/preflight/reference.npz'
SIXTH = E/'raw_ssl_next_20260913/ssl_runs/fold4/standard/oof.npz'
SCALE = np.array([0.9970446110788349, 0.987972130802936, 0.9892234872488541])
MEMBERS = {46: E/'phase2_pooling_20261020/arms/std', 47: E/'phase2_pooling_replication_20261026/f4_s47_std',
           48: E/'oof_recipeA_20261027/fold4/regression_std_s48'}
ANALYSIS = E/'recipeA_analysis_20261028'
OUT = ANALYSIS/'ensemble_vs_historical.json'
B, SEED = 10000, 20261028
TARGETS = ('CD', 'CV', 'HEX')


def mape(p, t):
    return np.abs(p-t)/t*100


def summary(p, t):
    v = mape(p, t).mean(0)
    return dict(CD=float(v[0]), CV=float(v[1]), HEX=float(v[2]), mean=float(v.mean()), n=int(len(t)))


def cluster_bootstrap(diff, groups, rng):
    uniq, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=diff, minlength=len(uniq))
    counts = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
    draws = rng.integers(0, len(uniq), size=(B, len(uniq)))
    return sums[draws].sum(1)/counts[draws].sum(1)


def main(allow_partial):
    analyses = sorted(ANALYSIS.glob('seed46_analysis_*.json'))
    final = [p for p in analyses if p.name.endswith('final.json')]
    gate_path = final[0] if final else (analyses[-1] if analyses else None)
    gates = json.loads(gate_path.read_text()) if gate_path else None
    use_tta = bool(gates and gates['tta_gate']['adopt'])
    cal = gates['calibration']['tta4' if use_tta else 'single'] if gates else None
    use_cal = bool(cal and cal['adopt'])
    scalars = np.array(cal['lofo_scalars_by_held_fold']['4']) if use_cal else np.ones(3)
    r = np.load(REF)
    o = np.load(SIXTH)
    assert (r['ids'] == o['ids']).all()
    ids, y, primary = r['ids'].astype(str), r['target'].astype(np.float64), r['primary']
    reference = (5*r['current5']+o['pred'].astype(np.float64))/6*SCALE
    ref_score = summary(reference[primary], y[primary])
    assert abs(ref_score['mean']-8.888057342844997) < 1e-6, ref_score
    members, used = [], []
    for seed, root in MEMBERS.items():
        p = root/'runs/fold4/cpdino_vitb/predictions_tta4.npz'
        if not p.exists():
            continue
        with np.load(p) as z:
            assert (z['ids'].astype(str) == ids).all()
            members.append((z['pred_tta4'] if use_tta else z['pred_single']).astype(np.float64))
        used.append(seed)
    if len(used) < 3 and not allow_partial:
        raise FileNotFoundError(f'members present: {used}')
    candidate = np.mean(members, 0)*scalars
    cand_score = summary(candidate[primary], y[primary])
    singles = {s: summary((m*scalars)[primary], y[primary])['mean'] for s, m in zip(used, members)}
    diff = mape(candidate[primary], y[primary]).mean(1)-mape(reference[primary], y[primary]).mean(1)
    groups = np.array([i.rsplit('-', 1)[0] for i in ids[primary]])
    boot = cluster_bootstrap(diff, groups, np.random.default_rng(SEED))
    decision = 'recipe A ensemble is submission 1' if cand_score['mean'] <= ref_score['mean'] else 'Phase 1 package is submission 1; recipe A ensemble is the submission-2 candidate'
    report = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), rows='fold4 primary matched', n=int(primary.sum()),
                  gates_source=str(gate_path), tta_used=use_tta, calibration_used=use_cal, scalars_fit_without_fold4=scalars.tolist(),
                  members_used=used, complete=len(used) == 3, reference=ref_score, candidate=cand_score,
                  member_singles_calibrated=singles, delta_candidate_minus_reference_pp=cand_score['mean']-ref_score['mean'],
                  per_target_delta_pp={k: cand_score[k]-ref_score[k] for k in TARGETS},
                  component_regression_over_0p05=[k for k in TARGETS if cand_score[k]-ref_score[k] > 0.05],
                  bootstrap=dict(ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], p_candidate_better=float((boot < 0).mean()),
                                 method='paired cluster bootstrap over id-prefix groups, 10,000 resamples'),
                  point_rule_decision=decision if len(used) == 3 else f'preview with {len(used)} members; no decision',
                  reference_note='historical six uses validation-selected checkpoints (optimistic reference); fixed Phase 1 scalars',
                  limitations=['fold4 is a reused development fold', 'patient independence unresolved; groups are id prefixes',
                               'candidate calibration scalars come from seed-46 OOF of folds 0-3 and are applied to seed-average predictions'])
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(members=used, tta=use_tta, cal=use_cal, reference=round(ref_score['mean'], 4), candidate=round(cand_score['mean'], 4),
                          delta=round(report['delta_candidate_minus_reference_pp'], 4), ci=report['bootstrap']['ci95'], p_better=report['bootstrap']['p_candidate_better'],
                          singles=singles, decision=report['point_rule_decision'])), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--allow-partial', action='store_true')
    main(ap.parse_args().allow_partial)
