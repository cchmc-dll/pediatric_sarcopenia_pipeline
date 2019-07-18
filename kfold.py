from argparse import ArgumentParser
from collections import namedtuple
from pathlib import Path

import numpy as np
import tables
from sklearn.model_selection import KFold, train_test_split


ThreewayKFold = namedtuple('ThreewayKFold', 'train_xs val_xs test_xs train_ys val_ys test_ys')


def main():
    args = parse_args()
    data_file = tables.open_file(args.input_file)

    kfold = kfold_with_train_val_and_test(
        xs=np.array(data_file.root.imdata),
        ys=np.array(data_file.root.truth)
    )

    kfold_directory = create_output_directory(
        input_file_path=Path(args.input_file),
        containing_dir_path=Path(args.output_dir)
    )

    save_kfold(kfold_directory, kfold)


    # for fold_index in range(len(kfold.train_xs)):



def parse_args():
    parser = ArgumentParser()

    parser.add_argument('input_file', help='Input .h5 file')
    parser.add_argument('output_dir', help='Output directory for kfold split h5 files')

    return parser.parse_args()


def kfold_with_train_val_and_test(xs, ys):
    kfold_indices = get_kfold_indices(xs, ys)

    train_xs = []
    val_xs = []
    test_xs = []

    train_ys = []
    val_ys = []
    test_ys = []

    for train, val, test in zip(*kfold_indices):
        train_xs.append(xs[train])
        val_xs.append(xs[val])
        test_xs.append(xs[test])

        train_ys.append(ys[train])
        val_ys.append(ys[val])
        test_ys.append(ys[test])

    return ThreewayKFold(train_xs, val_xs, test_xs, train_ys, val_ys, test_ys)


def get_kfold_indices(xs, ys):
    sets_of_training_train_indices = []
    sets_of_training_val_indices = []
    sets_of_test_indices = []

    kf = KFold(n_splits=5)
    for train_indices, test_indices in kf.split(xs, ys):
        train_train_indices, train_val_indices = train_test_split(train_indices, test_size=0.1)

        sets_of_training_train_indices.append(train_train_indices)
        sets_of_training_val_indices.append(train_val_indices)
        sets_of_test_indices.append(test_indices)

    return sets_of_training_train_indices, sets_of_training_val_indices, sets_of_test_indices


def create_output_directory(input_file_path, containing_dir_path):
    new_dir_name = input_file_path.stem
    new_output_directory = Path(containing_dir_path, new_dir_name)
    new_output_directory.mkdir()

    return new_output_directory


def save_kfold(directory, kfold):
    for fold_number in range(len(kfold.train_xs)):
        file_name = f'fold'


if __name__ == '__main__':
    main()
