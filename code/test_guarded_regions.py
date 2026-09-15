"""Synthetic-only tests; no dataset images need to leave the private workspace."""
import unittest
import numpy as np
from guarded_region_views import tile_quality, choose, native_origins, GuardedRegionViews
from fullframe_views import grid_origins


class Tests(unittest.TestCase):
    def test_blank_and_clipped_rejected(self):
        for value in (0,120,255):self.assertFalse(tile_quality(np.full((256,256),value,np.uint8))[2])

    def test_texture_accepted(self):
        raw=np.random.default_rng(4).integers(30,220,(256,256),dtype=np.uint8)
        self.assertTrue(tile_quality(raw)[2])
        raw[:128]=255
        self.assertFalse(tile_quality(raw)[2])

    def test_acceptable_priority_and_fixed_count(self):
        origins=grid_origins(972,1296); n=len(origins)
        scores=np.ones(n); scores[:8]=100
        acceptable=np.ones(n,bool);acceptable[:8]=False
        selected,ix=choose(origins,scores,np.ones(n),np.zeros(n),acceptable)
        self.assertTrue(acceptable[ix].all())
        self.assertEqual(len(set(map(tuple,selected))),8)

    def test_deterministic_fallback(self):
        origins=grid_origins(972,1296);n=len(origins)
        a=choose(origins,np.zeros(n),np.zeros(n),np.ones(n),np.zeros(n,bool))[0]
        b=choose(origins,np.zeros(n),np.zeros(n),np.ones(n),np.zeros(n,bool))[0]
        np.testing.assert_array_equal(a,b);self.assertEqual(len(set(map(tuple,a))),8)

    def test_native_determinism_bounds(self):
        x=np.random.default_rng(3).integers(20,220,(600,800),dtype=np.uint8)
        a,b=native_origins(x),native_origins(x)
        np.testing.assert_array_equal(a,b)
        self.assertTrue(((a>=0)&(a<=np.array([344,544]))).all())

    def test_jitter_fallback_same_rng_consumption(self):
        v=object.__new__(GuardedRegionViews)
        v.height=v.width=768
        v.images=np.zeros((1,768,768),np.uint8)
        v.selected=np.array([grid_origins(768,768)[:8]])
        np.random.seed(77)
        j=np.random.randint(8);np.random.randint(-32,33,size=2);expected=np.random.randint(100000)
        np.random.seed(77)
        np.testing.assert_array_equal(v.train_origin(0),v.selected[0,j])
        self.assertEqual(np.random.randint(100000),expected)


if __name__=="__main__":unittest.main()
