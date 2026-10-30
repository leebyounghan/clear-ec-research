"""Re-aggregate the recipe A seed-46 OOF with the fold2 v2 root (first fold2 root failed its own CPU gate).

Keeps the partial result as oof_result_partial_fold2_missing.json and writes oof_result.json
and oof_predictions.npz over all five folds using the supervisor's aggregate() unchanged.
"""
import importlib.util
import pathlib

spec = importlib.util.spec_from_file_location('oof', 'experiments/oof_recipeA_20261027/supervisor.py')
oof = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oof)
R = oof.R
oof.EXISTING[2] = str(R/'fold2'/'regression_std_s46_v2')
partial = R/'oof_result_partial_fold2_missing.json'
if (R/'oof_result.json').exists() and not partial.exists():
    (R/'oof_result.json').rename(partial)
    (R/'oof_predictions.npz').rename(R/'oof_predictions_partial_fold2_missing.npz')
oof.aggregate()
print(pathlib.Path(R/'oof_result.json').read_text()[:600])
