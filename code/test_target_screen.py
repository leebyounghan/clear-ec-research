"""Synthetic data only; target isolation and fixed gates."""
import unittest
import numpy as np
from target_screen_core import targets_for_training, summarize_target_shift, decision


class Tests(unittest.TestCase):
    def setUp(self):
        self.y = np.array([[1,2,3],[3,4,5],[9,8,7],[11,12,13]], np.float32)
        self.slides = np.array(['a','a','a','b'])
        self.tr = np.array([0,1,3])

    def test_raw_identity_and_copy(self):
        actual=targets_for_training(self.y,self.slides,self.tr,'raw')
        np.testing.assert_array_equal(actual,self.y)
        self.assertFalse(np.shares_memory(actual,self.y))

    def test_smoothing_train_only(self):
        actual=targets_for_training(self.y,self.slides,self.tr,'slidesmooth')
        np.testing.assert_array_equal(actual[:2],[[2,3,4],[2,3,4]])
        np.testing.assert_array_equal(actual[2:],self.y[2:])
        altered=self.y.copy(); altered[2]=100000
        other=targets_for_training(altered,self.slides,self.tr,'slidesmooth')
        np.testing.assert_array_equal(actual[self.tr],other[self.tr])

    def test_bad_labels_fail(self):
        broken=self.y.copy(); broken[0,0]=0
        with self.assertRaises(ValueError):
            targets_for_training(broken,self.slides,self.tr,'raw')

    def test_no_change_detected(self):
        y=np.ones((4,3),np.float32)
        self.assertEqual(summarize_target_shift(y,self.slides,self.tr)['changed_rows'],0)

    def test_gate_no_single_fold_rescue(self):
        aggregate=dict(standalone_vs_matched=-.1,replace6_vs_current6=-.03)
        good=dict(standalone_vs_matched=-.1,replace6_vs_current6=-.03,replace6_vs_matched=-.02)
        bad=dict(good,replace6_vs_current6=.001)
        self.assertTrue(decision(aggregate,[good,good])['passes_ensemble_screen'])
        self.assertFalse(decision(aggregate,[good,bad])['passes_ensemble_screen'])
        self.assertFalse(decision(aggregate,[good,good])['deployment_approved'])


if __name__=='__main__':
    unittest.main()
