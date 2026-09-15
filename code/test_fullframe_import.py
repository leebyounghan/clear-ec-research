import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
import numpy as np
import SimpleITK as sitk
from prepare_fullframe_cache import safe_member, sha256


class ImportTests(unittest.TestCase):
    def test_archive_paths(self):
        self.assertEqual(str(safe_member('train_mha/2020-0823 OSCN.mha')),
                         'train_mha/2020-0823 OSCN.mha')
        self.assertEqual(str(safe_member('final_train_ids.csv')), 'final_train_ids.csv')
        for value in ('/train_mha/x.mha', '../x.mha', 'train_mha/../x.mha',
                      'other/x.mha', 'train_mha\\x.mha'):
            with self.assertRaises(ValueError):
                safe_member(value)

    def test_complete_import_and_idempotence(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            source = root / 'source' / 'train_mha'; source.mkdir(parents=True)
            reference = root / 'reference'; reference.mkdir()
            out = root / 'output'
            pixels = np.random.default_rng(1).integers(0, 256, (972, 1296), dtype=np.uint8)
            raw = source / 'sample image.mha'
            sitk.WriteImage(sitk.GetImageFromArray(pixels), str(raw))
            archive_path = root / 'raw.tar.gz'
            r0, c0 = (972 - 512) // 2, (1296 - 512) // 2
            pixels[r0:r0+512, c0:c0+512].tofile(reference / 'img_native_512.dat')
            np.savez(reference / 'cache_native_meta_512.npz', ids=np.array(['sample image']),
                     Y=np.array([[2500, .3, .5]]), N=1, SZ=512)
            (reference / 'final_train_ids.csv').write_text('ID,CD,CV,HEX,slide_id\nsample image,2500,.3,.5,s1\n')
            with tarfile.open(archive_path, 'w:gz') as archive:
                archive.add(source, arcname='train_mha')
                archive.add(reference / 'final_train_ids.csv', arcname='final_train_ids.csv')
            command = [sys.executable, str(Path(__file__).with_name('prepare_fullframe_cache.py')),
                       '--archive', str(archive_path), '--sha256', sha256(archive_path),
                       '--reference', str(reference), '--out', str(out)]
            for _ in range(2):
                result = subprocess.run(command, capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            manifest = json.loads((out / 'fullframe_manifest.json').read_text())
            self.assertTrue(manifest['central_pixels_exact_match'])
            self.assertTrue(manifest['archive_labels_match_cache'])
            self.assertEqual(manifest['matched_cache_rows'], 1)
            restored = np.memmap(out / 'img_full_raw.dat', dtype=np.uint8, mode='r', shape=(972,1296))
            np.testing.assert_array_equal(restored, pixels)


if __name__ == '__main__':
    unittest.main()
