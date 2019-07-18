import numpy as np

from unittest import TestCase

from kfold import get_kfold_indices, kfold_with_train_val_and_test


class TestKfold(TestCase):
    def setUp(self) -> None:
        self.x = np.arange(100, 110)
        self.y = np.arange(200, 210)

    def test_get_kfold_indices_should_split_data_into_3_sets_of_5_folds_by_default(self):
        xs, vals, ys = get_kfold_indices(self.x, self.y)

        self.assertEqual(len(xs), 5)
        self.assertEqual(len(vals), 5)
        self.assertEqual(len(ys), 5)

    def test_get_kfold_indices_should_not_overlap(self):
        kfold_indices = get_kfold_indices(self.x, self.y)

        for train, val, test in zip(*kfold_indices):
            self.assertFalse(any(i in val for i in train))
            self.assertFalse(any(i in test for i in train))

            self.assertFalse(any(i in train for i in val))
            self.assertFalse(any(i in test for i in val))

            self.assertFalse(any(i in train for i in test))
            self.assertFalse(any(i in val for i in test))

    def test_get_kfold_indices_should_error_if_input_arrays_lengths_not_equal(self):
        self.assertRaises(ValueError, get_kfold_indices, xs=[1], ys=[2, 3])

    def test_kfold_with_train_val_and_test_should_return_values(self):
        results = kfold_with_train_val_and_test(self.x, self.y)

        for train_xs, val_xs, test_xs, train_ys, val_ys, test_ys in zip(*results):
            self.assertTrue(all(t in self.x for t in train_xs))
            self.assertTrue(all(v in self.x for v in val_xs))
            self.assertTrue(all(t in self.x for t in test_xs))

            self.assertTrue(all(t in self.y for t in train_ys))
            self.assertTrue(all(v in self.y for v in val_ys))
            self.assertTrue(all(t in self.y for t in test_ys))
