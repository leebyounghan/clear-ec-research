"""CPU contract tests; execute on Kubernetes, not the laptop."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np

SOURCE = Path('experiments/raw_ssl_next_20260913/source/next_candidate_20260913.py')
spec = importlib.util.spec_from_file_location('candidate', SOURCE)
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)


class Contracts(unittest.TestCase):
    def test_checkpoint_interface(self):
        shared = SimpleNamespace(LM=np.array([1., 2., 3.]), LS=np.ones(3))
        cfg = candidate.config_for(shared, 5215, False, train_rows=8998)
        self.assertEqual(cfg['target_policy'], 'raw')
        self.assertEqual(cfg['region_policy'], 'full_quality')
        self.assertEqual(cfg['k_crops'], 2)
        self.assertEqual(cfg['seed'], 77)
        self.assertEqual(cfg['pool'], 'mean')
        self.assertEqual(cfg['train_rows'], 8998)
        self.assertFalse(cfg['smoke'])

    def test_raw_labels_not_smoothed(self):
        tree = ast.parse(SOURCE.read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'raw_train')
        keys = [n.slice.value for n in ast.walk(fn) if isinstance(n, ast.Subscript)
                and isinstance(n.slice, ast.Constant)]
        self.assertIn('Y', keys)
        self.assertNotIn('Y_train', keys)

    def test_fixed_two_rate_screen(self):
        self.assertEqual(candidate.FOLDS, (2, 3, 4))
        self.assertEqual(candidate.RATES, {'standard': 1e-5, 'half_lr': 5e-6})
        self.assertEqual(candidate.source_for(2), candidate.REPLICA)
        self.assertEqual(candidate.source_for(4), candidate.FOLD4)

    def test_manifest_and_protected_inventory(self):
        m = candidate.verify()
        self.assertEqual(m['raw_updates'], 5215)
        self.assertTrue(m['no_automatic_ssl_promotion'])
        self.assertTrue(m['no_automatic_submission'])
        self.assertEqual(len(m['old_members']), 8)
        self.assertIn(candidate.RAW_NAME, m['old_members'])
        for row in m['folds'].values():
            self.assertEqual(row['slide_and_exact_duplicate_overlap'], 0)

    def test_venv_path_preserved(self):
        text = SOURCE.read_text()
        self.assertIn("os.path.abspath(manifest['runtime_python'])", text)
        self.assertNotIn("Path(manifest['runtime_python']).resolve()", text)


if __name__ == '__main__':
    unittest.main(verbosity=2)
