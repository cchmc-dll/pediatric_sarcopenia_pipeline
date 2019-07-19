import numpy as np

from unittest import TestCase

from kfold import get_kfold


class TestKfold(TestCase):
    def setUp(self) -> None:
        self.x = np.arange(100, 110)
        self.y = np.arange(200, 210)

    def test_get_kfold_indices_should_split_data_into_3_sets_of_5_folds_by_default(self):
        xs, vals, ys = get_kfold(self.x, self.y)

        self.assertEqual(len(xs), 5)
        self.assertEqual(len(vals), 5)
        self.assertEqual(len(ys), 5)

    def test_get_kfold_indices_should_not_overlap(self):
        kfold_indices = get_kfold(self.x, self.y)

        for train, val, test in zip(*kfold_indices):
            self.assertFalse(any(i in val for i in train))
            self.assertFalse(any(i in test for i in train))

            self.assertFalse(any(i in train for i in val))
            self.assertFalse(any(i in test for i in val))

            self.assertFalse(any(i in train for i in test))
            self.assertFalse(any(i in val for i in test))

    def test_get_kfold_indices_should_error_if_input_arrays_lengths_not_equal(self):
        self.assertRaises(ValueError, get_kfold, xs=[1], ys=[2, 3])
