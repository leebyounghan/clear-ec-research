"""Synthetic, dependency-free tests; no challenge images or CUDA required."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import predict_image as runner


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='lamda-wrapper-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / 'synthetic case.mha'
        self.image.write_bytes(b'synthetic input bytes; not a clinical image')
        self.model = self.root / 'model'
        self.model.mkdir()
        members = json.loads((runner.ROOT / 'provenance/model_archive.json').read_text())['members']
        for name in members:
            (self.model / name).write_bytes(b'synthetic fixture')
        self.output = self.root / 'result'

    def test_valid_paths(self):
        runner.validate_paths(self.image, self.model, self.output)

    def test_missing_image(self):
        with self.assertRaises(ValueError):
            runner.validate_paths(self.root / 'absent.mha', self.model, self.output)

    def test_missing_checkpoint(self):
        (self.model / 'cpsam_v2').unlink()
        with self.assertRaises(FileNotFoundError):
            runner.validate_paths(self.image, self.model, self.output)

    def test_existing_results_preserved(self):
        self.output.mkdir()
        result = self.output / 'keep.json'
        result.write_text('42')
        with self.assertRaises(FileExistsError):
            runner.validate_paths(self.image, self.model, self.output)
        self.assertEqual(result.read_text(), '42')

    def test_output_file_rejected(self):
        self.output.write_text('keep')
        with self.assertRaises(FileExistsError):
            runner.validate_paths(self.image, self.model, self.output)

    def test_output_symlink_rejected(self):
        self.output.symlink_to(self.root / 'missing')
        with self.assertRaises(FileExistsError):
            runner.validate_paths(self.image, self.model, self.output)

    def test_gpu_allocation_preserved(self):
        self.assertEqual(runner.select_device(0, {'CUDA_VISIBLE_DEVICES': '2,3'}), '2')
        self.assertEqual(runner.select_device(1, {'CUDA_VISIBLE_DEVICES': 'GPU-a,GPU-b'}), 'GPU-b')
        self.assertEqual(runner.select_device(2, {}), '2')
        for value in ('', '-1', '0'):
            with self.assertRaises(ValueError):
                runner.select_device(1, {'CUDA_VISIBLE_DEVICES': value})

    def test_wrapper_preserves_bytes_and_frozen_options(self):
        temporary_inputs = []
        def fake_run(command, **kwargs):
            self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '2')
            if command[1].endswith('/app/inference.py'):
                self.assertEqual(command[-1], '--fp16')
                self.assertEqual(command[command.index('--config') + 1], 'candidate6_full_quality.json')
                source = Path(command[command.index('--input') + 1])
                temporary_inputs.append(source)
                self.assertEqual((source / 'images/corneal-specular-microscopy-image/case.mha').read_bytes(), self.image.read_bytes())
                self.assertEqual(json.loads((source / 'inputs.json').read_text())[0]['socket']['slug'],
                                 'corneal-specular-microscopy-image')
                self.output.mkdir()
                for name in runner.OUTPUTS:
                    (self.output / f'{name}.json').write_text('1.0')
        argv = ['predict_image.py', '--image', str(self.image), '--model', str(self.model),
                '--output', str(self.output), '--gpu', '0']
        with patch('sys.argv', argv), patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '2,3'}), \
                patch.object(runner.subprocess, 'run', side_effect=fake_run) as run, \
                contextlib.redirect_stdout(io.StringIO()):
            runner.main()
        self.assertEqual(run.call_count, 3)
        self.assertTrue(all(not p.exists() for p in temporary_inputs))
        self.assertEqual(len(list(self.output.glob('*.json'))), 3)


if __name__ == '__main__':
    unittest.main()
