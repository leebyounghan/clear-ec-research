"""Synthetic tests only; run on Kubernetes, not on the local Mac."""
import unittest
import numpy as np
from overnight_target import connected_groups, clean_split, gate, targets_for_training


class Tests(unittest.TestCase):
    def test_transitive_duplicate_groups(self):
        slides=np.array(['a','b','b','c','d'])
        groups=connected_groups(slides,['x','x','y','y','z'])
        self.assertEqual(len(set(groups[:4])),1)
        self.assertNotEqual(groups[0],groups[4])

    def test_clean_split_primary(self):
        groups=np.array(['a','b','b','c','d'])
        tr,va=clean_split(np.array([0,1,4]),np.array([2,3]),groups)
        np.testing.assert_array_equal(tr,[0,4]);np.testing.assert_array_equal(va,[3])

    def test_row_overlap_fails(self):
        with self.assertRaises(AssertionError):clean_split(np.array([0,1]),np.array([1]),np.array(['a','b']))

    def test_targets_exclude_validation(self):
        y=np.array([[1,2,3],[3,4,5],[999,999,999]],np.float32)
        out=targets_for_training(y,np.array(['s','s','s']),np.array([0,1]),'slidesmooth')
        np.testing.assert_array_equal(out[:2],[[2,3,4],[2,3,4]])
        np.testing.assert_array_equal(out[2],y[2])
        np.testing.assert_array_equal(targets_for_training(y,np.array(['s','s','s']),np.array([0,1]),'raw'),y)

    def test_gate_requires_two_folds(self):
        self.assertTrue(gate(dict(aggregate=-.03,per_fold=[-.01,-.05])))
        self.assertFalse(gate(dict(aggregate=-.03,per_fold=[.01,-.07])))
        self.assertFalse(gate(dict(aggregate=-.01,per_fold=[-.01,-.01])))
        self.assertFalse(gate(dict(aggregate=-.03,per_fold=[-.03])))


if __name__=='__main__':unittest.main()
