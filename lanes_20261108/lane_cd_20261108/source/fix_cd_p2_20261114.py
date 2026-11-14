"""Correct the fields that build_cd_p2_20261114.py copied from the tier-p (t1) file: they described t1, not p2."""
import json, hashlib
R = 'experiments/lane_cd_20261108/reads/'; p = R + 'freeze_cd_params_p2_20261114.json'; d = json.load(open(p))
inc = json.load(open(R + 'cd_tier_inc_20261112.json'))['strict_five_fold_cd']
d['incumbent_D_CD_001_strict_same_run'] = inc; d['delta_vs_incumbent'] = round(d['strict_five_fold_cd'] - inc, 5)
d['delta_vs_best_pipeline_as_computed'] = round(d['strict_five_fold_cd'] - d['best_pipeline_cd_as_computed'], 4)
d['delta_ci95_donor_bootstrap'] = None; d['information_only'] = False
d['per_fold_delta_note'] = 'per_fold_delta is against tier p (t1), fold by fold'
d['interval_note'] = 'interval against tier p from the drop ladder (O-CD-DROPLADDER): strict -0.0014 (CI -0.005 to +0.002); 10-fold stack read -0.0015 (CI -0.004 to +0.001)'
json.dump(d, open(p, 'w'), indent=1); b = open(p, 'rb').read()
print(json.dumps(dict(md5=hashlib.md5(b).hexdigest(), sha256=hashlib.sha256(b).hexdigest(), strict=d['strict_five_fold_cd'], tier_p=d['tier_p_cd_t1_strict'], delta_vs_tier_p=d['delta_vs_tier_p'],
                      inc=inc, delta_inc=d['delta_vs_incumbent'], per_fold=d['per_fold_delta'], L5=d['L5'], weights=d['deploy']['weights'], files=len(d['model_files']), cd_only=d['model_bytes_packaged_cd_only'])))
