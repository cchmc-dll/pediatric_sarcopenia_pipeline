import pickle
import sys
from argparse import ArgumentParser
from functools import partialmethod
from pathlib import Path

import numpy as np


class Fold:
    def __init__(self, root_path, fold_index):
        self.root_path = root_path
        self.fold_index = fold_index

    def load_indices_for_category(self, category):
        path = Path(self.root_path,
                    "fold_{}_{}.pkl".format(self.fold_index, category))
        with open(path, "rb") as f:
            return pickle.load(f)

    get_train_indices = partialmethod(load_indices_for_category, "train")
    get_val_indices = partialmethod(load_indices_for_category, "val")
    get_test_indices = partialmethod(load_indices_for_category, "test")


def main(argv):
    args = parse_args(argv)
    new_folds = substitute_folds(args)
    save_folds(args, new_folds)
    return new_folds
    pass


def parse_args(argv):
    parser = ArgumentParser()
    parser.add_argument("input_directory",
                        help="Directory with fold pickle files")
    parser.add_argument("indices",
                        help="comma separated list of indices to swap")
    parser.add_argument("target_indices",
                        help="comma separated list of indices to move from set")
    parser.add_argument("output_directory", help="where the new folds will go")

    return parser.parse_args(argv)


def substitute_folds(args):
    folds = [Fold(args.input_directory, i) for i in range(5)]

    indices = [int(i) for i in args.indices.split(',')]
    target_indices = [int(i) for i in args.target_indices.split(',')]

    result_folds = []

    for fold in folds:
        test_indices = fold.get_test_indices()
        train_indices = fold.get_train_indices()
        val_indices = fold.get_val_indices()

        for new_index, target_index in zip(indices, target_indices):
            test_indices = np.where(test_indices == target_index, new_index,
                                    test_indices)
            train_indices = np.where(train_indices == new_index, target_index,
                                     train_indices)
            val_indices = np.where(val_indices == new_index, target_index,
                                   val_indices)
        result_folds.append(
            {
                "train": train_indices,
                "val": val_indices,
                "test": test_indices,
            }
        )

    return result_folds


def save_folds(args, new_folds):
    output_dir = Path(args.output_directory)
    for index, fold in enumerate(new_folds):
        for split_name in fold.keys():
            split_path = Path(output_dir, f"fold_{index}_{split_name}.pkl")

            with open(split_path, "wb") as f:
                pickle.dump(fold[split_name], f)



if __name__ == "__main__":
    main(sys.argv[1:])
