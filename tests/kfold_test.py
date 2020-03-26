import numpy as np

from unittest import TestCase

from kfold import get_kfold, get_training_and_test_split


class TestKfold(TestCase):
    def setUp(self) -> None:
        self.x = np.arange(100, 110)
        self.y = np.arange(200, 210)
        self.subject_ids = np.arange(300, 310)

    def test_get_kfold_indices_should_split_data_into_3_sets_of_5_folds_by_default(self):
        kfold = get_kfold(
            xs=self.x,
            ys=self.y,
            subject_ids=self.subject_ids,
            train_test_split=self._get_train_test_split(),
        )

        self.assertEqual(len(kfold.train_indices), 5)
        self.assertEqual(len(kfold.val_indices), 5)
        # should only be 1 test set with length 20%
        self.assertEqual(len(kfold.test_indices), 5)

    def _get_train_test_split(self):
        return get_training_and_test_split(xs=self.x, existing_test_set_file=None)

    def test_get_kfold_indices_should_not_overlap(self):
        kfold_indices = get_kfold(
            xs=self.x,
            ys=self.y,
            subject_ids=self.subject_ids,
            train_test_split=self._get_train_test_split()
        )

        for train, val, test, _, _, _ in zip(*kfold_indices):
            self.assertFalse(any(i in val for i in train))
            self.assertFalse(any(i in test for i in train))

            self.assertFalse(any(i in train for i in val))
            self.assertFalse(any(i in test for i in val))

            self.assertFalse(any(i in train for i in test))
            self.assertFalse(any(i in val for i in test))

    def test_get_kfold_indices_should_error_if_input_arrays_lengths_not_equal(self):
        self.assertRaises(
            AssertionError,
            get_kfold,
            xs=np.array([1]),
            ys=np.array([2, 3]),
            subject_ids=np.array([4, 5, 6]),
            train_test_split=get_training_and_test_split(
                xs=self.x,
                existing_test_set_file=None,
            ),
        )
