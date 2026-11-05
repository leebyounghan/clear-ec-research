import ast
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from native_region_views import NativeRegionViews,POLICIES,diverse_origins
from fullframe_views import prepare
from train_native_external_ssl import native_crop,block_mask,SOURCES
from run_region_search import job
from audit_region_search import improvement_gate


class Tests(unittest.TestCase):
    def test_diversity_and_anchors(self):
        origins=np.array([(r,c) for r in range(0,717,143) for c in range(0,1041,208)])
        scores=np.ones(len(origins))
        a=diverse_origins(origins,scores,8)
        np.testing.assert_array_equal(a,diverse_origins(origins,scores,8))
        self.assertEqual(len(np.unique(a,axis=0)),8)
        b=diverse_origins(origins,scores,2,anchors=a[:3])
        self.assertFalse(any(np.array_equal(p,q) for p in b for q in a[:3]))
        with self.assertRaises(ValueError): diverse_origins(origins,np.full(len(origins),np.nan))

    def test_native_full_resolution_views(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); pixels=np.random.default_rng(9).integers(0,256,(1,972,1296),dtype=np.uint8)
            pixels.tofile(root/"img_full_raw.dat")
            (root/"fullframe_manifest.json").write_text(json.dumps(dict(cache_shape=list(pixels.shape),archive_sha256="test",central_pixels_exact_match=True)))
            np.savez(root/"fullframe_meta.npz",ids=np.array(["x"]));prepare(root)
            for policy in POLICIES:
                views=NativeRegionViews(pixels[:,230:742,392:904],["x"],policy,raw_root=root)
                self.assertEqual(views.train(0).shape,(2,3,256,256))
                self.assertEqual(views.evaluate(0,1).shape,(40,3,256,256))
                origins=views.eval_origins_for(0);self.assertEqual(len(np.unique(origins,axis=0)),5)
                views.normalize=lambda raw:raw.astype(np.float32)
                value=views.evaluate(0,0)[0,0];r,c=origins[0]
                np.testing.assert_array_equal(value,pixels[0,r:r+256,c:c+256])
                for _ in range(50):
                    r,c=views.train_origin(0)
                    self.assertTrue(0<=r<=716 and 0<=c<=1040)

    def test_external_native_geometry_and_mask(self):
        image=np.tile(np.arange(300,dtype=np.uint16)%2,(400,1)).astype(np.uint8)*255
        for _ in range(10):
            self.assertEqual(set(np.unique(native_crop(image))),{0,255})
            m=block_mask(); self.assertGreaterEqual(m.mean(),.4);self.assertLess(m.mean(),.72)
        self.assertEqual(native_crop(np.zeros((100,80),np.uint8)).shape,(256,256))
        self.assertEqual(len(SOURCES),2)
        self.assertFalse(any("clear" in s for s in SOURCES))

    def test_model_and_folds_preserved(self):
        here=Path(__file__).parent
        a=ast.parse((here/"train_method_h200.py").read_text());b=ast.parse((here/"train_region_method.py").read_text())
        for name in ("ScalarNet","make_group_folds","seed_everything","normalize"):
            x=next(n for n in a.body if getattr(n,"name",None)==name)
            y=next(n for n in b.body if getattr(n,"name",None)==name)
            self.assertEqual(ast.dump(x),ast.dump(y))

    def test_frozen_compute_and_selection(self):
        for policy in POLICIES:
            for fold in (0,1):
                j=job(policy,fold);e=j["env"]
                self.assertEqual((e["EPOCHS"],e["BS"],e["K_CROPS"]),(35,48,2))
                self.assertEqual(e["MAX_TRAIN_STEPS"],0);self.assertEqual(e["ENCODER_INIT"],"")
        self.assertFalse(improvement_gate(dict(delta_add6=.001,fold_delta_add6=[.001,.001],delta_replace4=.003,fold_delta_replace4=[.003,.003])))
        self.assertTrue(improvement_gate(dict(delta_add6=-.02,fold_delta_add6=[-.01,-.03],delta_replace4=.003,fold_delta_replace4=[.003,.003])))
        self.assertFalse(improvement_gate(dict(delta_add6=-.02,fold_delta_add6=[-.05,.01],delta_replace4=.003,fold_delta_replace4=[.003,.003])))


if __name__=="__main__":unittest.main()
