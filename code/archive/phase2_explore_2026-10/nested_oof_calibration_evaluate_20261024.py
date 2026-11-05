"""One fixed prospective four-method fit. All outer results reported without tuning."""
import sys,json,hashlib,datetime
from pathlib import Path
import numpy as np
I=Path(sys.argv[1]);O=Path(sys.argv[2]);sys.path.insert(0,str(I))
from nested_oof_calibration_20261023 import METHODS,fit,predict,tests
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
a=json.loads((I/'inputs_audit.json').read_text());assert a['status']=='PASS'
assert sha(I/'inputs.npz')==a['inputs_sha256']
assert sha(I/'nested_oof_calibration_20261023.py')=='333a80395fd98f2c8d57d6b4583903f29db7193edcd2b0c92866fe4c1af0a1bc'
z=np.load(I/'inputs.npz');assert len(z['fit_ids'])==7150 and len(z['outer_ids'])==1790
assert not set(z['fit_ids'])&set(z['outer_ids']);assert len(set(z['fit_ids']))==7150
assert tests()['status']=='PASS';O.mkdir(exist_ok=False)
models={};out={};scores={}
# Freeze all fitted models before reading any outer targets.
for method in METHODS:
 m=fit(method,z['fit_pred'],z['fit_target'],z['fit_quality'],z['fit_ids'].tolist(),z['outer_ids'].tolist())
 models[method]=m
with (O/'models.json').open('x') as f:json.dump(models,f,indent=2)
reloaded=json.loads((O/'models.json').read_text())
for method in METHODS:
 p=predict(reloaded[method],z['outer_pred'],z['outer_quality'])
 np.testing.assert_array_equal(p,predict(models[method],z['outer_pred'],z['outer_quality']))
 out[method]=p
 if method=='identity':np.testing.assert_array_equal(p,z['outer_pred'])
 v=(np.abs(p-z['outer_target'].astype(np.float64))/z['outer_target']).mean(0)*100
 scores[method]=dict(zip(['CD','CV','HEX'],v.tolist()));scores[method]['mean']=float(v.mean())
np.savez_compressed(O/'outer_predictions.npz',indices=z['outer_indices'],ids=z['outer_ids'],target=z['outer_target'],**out)
result=dict(status='fixed_four_method_outer_evaluation_complete',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),fit_n=7150,outer_n=1790,outer_fold=4,scores=scores,delta_mean_vs_identity={k:v['mean']-scores['identity']['mean'] for k,v in scores.items()},inputs_sha256=sha(I/'inputs.npz'),input_audit_sha256=sha(I/'inputs_audit.json'),source_sha256=sha(__file__),models_sha256=sha(O/'models.json'),predictions_sha256=sha(O/'outer_predictions.npz'),serialization_exact=True,outer_label_tuning=False,all_prospective_methods_reported=True,raw_base='SSL5 mean pooling terminal20',promotion=False,limitations=['Historically reused development fold; not independent generalization or official score','Single outer-fold pilot; additional outer confirmation required for promotion','Patient identity and near-duplicate independence unresolved','Calibration results separate from raw single-model and ensemble results'])
with (O/'result.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
