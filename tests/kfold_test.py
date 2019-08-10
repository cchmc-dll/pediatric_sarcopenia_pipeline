import numpy as np

from unittest import TestCase

from kfold import get_kfold


class TestKfold(TestCase):
    def setUp(self) -> None:
        self.x = np.arange(100, 110)
        self.y = np.arange(200, 210)
        self.subject_ids = np.arange(300, 310)

    def test_get_kfold_indices_should_split_data_into_3_sets_of_5_folds_by_default(self):
        kfold = get_kfold(xs=self.x, ys=self.y, subject_ids=self.subject_ids)

        self.assertEqual(len(kfold.train_indices), 5)
        self.assertEqual(len(kfold.val_indices), 5)
        self.assertEqual(len(kfold.test_indices), 5)

    def test_get_kfold_indices_should_not_overlap(self):
        kfold_indices = get_kfold(xs=self.x, ys=self.y, subject_ids=self.subject_ids)

        for train, val, test, _, _, _ in zip(*kfold_indices):
            self.assertFalse(any(i in val for i in train))
            self.assertFalse(any(i in test for i in train))

            self.assertFalse(any(i in train for i in val))
            self.assertFalse(any(i in test for i in val))

            self.assertFalse(any(i in train for i in test))
            self.assertFalse(any(i in val for i in test))

    def test_get_kfold_indices_should_error_if_input_arrays_lengths_not_equal(self):
        self.assertRaises(ValueError, get_kfold, xs=[1], ys=[2, 3], subject_ids=[4, 5, 6])
