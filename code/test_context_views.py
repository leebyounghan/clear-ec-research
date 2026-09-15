import ast
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from torch import nn

from fullframe_views import ContextViews, prepare
from run_context_screen import jobs, submission_reserved


class ContextTests(unittest.TestCase):
    def test_no_competing_gpu_launches(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            self.assertIsNone(submission_reserved(root))
            path=root/'status.json'
            path.write_text(json.dumps(dict(pending=['next'],active={'0':{}},done=[])))
            self.assertIsNone(submission_reserved(root))
            path.write_text(json.dumps(dict(pending=[],active={'0':{},'3':{}},done=['one'])))
            self.assertEqual(submission_reserved(root),{0,3})
            self.assertIsNone(submission_reserved(root,now=path.stat().st_mtime+121))
            (root/'complete.json').write_text('{}')
            self.assertEqual(submission_reserved(root),set())

    def test_central_shapes_and_augmentation(self):
        pixels=np.random.default_rng(4).integers(0,256,(1,512,512),dtype=np.uint8)
        for two in [False,True]:
            v=ContextViews(pixels,['x'],two_views=two)
            k=2 if two else 1
            self.assertEqual(v.train(0).shape,(k,3,256,256))
            arrays=v.evaluate(0,True)
            self.assertEqual(arrays.shape,(40,k,3,256,256))
            self.assertEqual(arrays.dtype,np.float32)
            self.assertTrue(np.isfinite(arrays).all())
            np.testing.assert_array_equal(arrays[1,0,0],np.rot90(arrays[0,0,0]))

    def test_full_cache_and_quality_shapes(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            pixels=np.random.default_rng(5).integers(0,256,(1,972,1296),dtype=np.uint8)
            pixels.tofile(root/'img_full_raw.dat')
            (root/'fullframe_manifest.json').write_text(json.dumps(dict(cache_shape=list(pixels.shape),
                archive_sha256='test-only',central_pixels_exact_match=True)))
            np.savez(root/'fullframe_meta.npz',ids=np.array(['x']))
            prepare(root)
            for quality in ['uniform','focus']:
                v=ContextViews(pixels[:,230:742,392:904],['x'],scope='full',quality=quality,raw_root=root)
                self.assertEqual(v.train(0).shape,(2,3,256,256))
                self.assertEqual(v.evaluate(0,False).shape,(5,2,3,256,256))
                self.assertTrue(np.isfinite(v.evaluate(0,False)).all())

    def test_model_and_fold_code_unchanged(self):
        here=Path(__file__).parent
        originals=ast.parse((here/'train_oof_h200.py').read_text())
        context=ast.parse((here/'train_context_oof.py').read_text())
        for name in ['ScalarNet','make_group_folds','seed_everything']:
            a=next(n for n in originals.body if getattr(n,'name',None)==name)
            b=next(n for n in context.body if getattr(n,'name',None)==name)
            self.assertEqual(ast.dump(a),ast.dump(b))
        klass=next(n for n in context.body if getattr(n,'name',None)=='ScalarNet')
        scope={'nn':nn,'torch':torch,'BAG_POOL':'attn'}
        exec(compile(ast.Module(body=[klass],type_ignores=[]),'<test_model>','exec'),scope)
        class Dummy(nn.Module):
            def forward(self,x):
                return nn.functional.adaptive_avg_pool2d(x.mean(1,keepdim=True),(2,2)).expand(-1,256,-1,-1)
        model=scope['ScalarNet'](Dummy(),'mean')
        self.assertEqual(tuple(model(torch.ones(2,2,3,256,256)).shape),(2,3))

    def test_bounded_jobs(self):
        planned=jobs()
        self.assertEqual(len(planned),8)
        self.assertEqual(len({j['run'] for j in planned}),8)
        for j in planned:
            self.assertEqual(j['env']['BS'],24)
            self.assertEqual(j['env']['EPOCHS'],25)
            self.assertEqual(j['env']['MAX_TRAIN_STEPS'],0)
            self.assertIn(j['fold'],[0,1])


if __name__=='__main__': unittest.main()
