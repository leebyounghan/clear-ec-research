"""CPU safety/import tests; no data or historical run directories required."""
import csv
import os
from pathlib import Path
import tempfile
import unittest

import workspace
import train_ssl_raw


class PortableTests(unittest.TestCase):
    def test_imports_outside_repository(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous = Path.cwd()
            try:
                os.chdir(tmp)
                modules = train_ssl_raw.load_api()
                self.assertFalse(Path('experiments').exists())
            finally:
                os.chdir(previous)
            self.assertEqual(len(modules), 4)
            self.assertEqual(len(train_ssl_raw.source_inventory()), 8)

    def test_existing_workspace_never_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            with self.assertRaises(FileExistsError):
                workspace.initialize(path, path, path, path)

    def test_small_dataset_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = root/'labels.csv'
            with labels.open('w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['ID', 'slide_id', 'CD', 'CV', 'HEX'])
                w.writerow(['case', 'slide', 2000, .3, .5])
            (root/'case.mha').touch()
            with self.assertRaises(ValueError):
                workspace.validate_inputs(root, labels)

    def test_existing_output_never_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError):
                train_ssl_raw.train(Path(tmp), Path(tmp))


if __name__ == '__main__':
    unittest.main()
