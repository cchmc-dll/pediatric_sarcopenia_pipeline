import os
from argparse import ArgumentParser
from collections import namedtuple
from pathlib import Path

import numpy as np
import tables
from sklearn.model_selection import KFold, train_test_split

from unet3d.utils import pickle_dump

ThreewayKFold = namedtuple('ThreewayKFold', 'train_indices val_indices test_indices')


def main():
    args = parse_args()

    with tables.open_file(args.input_file) as data_file:
        kfold_indices = get_kfold_indices(
            xs=np.array(data_file.root.imdata),
            ys=np.array(data_file.root.truth)
        )

    kfold_directory = create_output_directory(
        input_file_path=Path(args.input_file),
        containing_dir_path=Path(args.output_dir)
    )

    save_kfold_indices(kfold_directory, kfold_indices)


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('input_file', help='Input .h5 file')
    parser.add_argument('output_dir', help='Directory where dir of kfold files will be created')

    return parser.parse_args()


def get_kfold_indices(xs, ys):
    sets_of_training_train_indices = []
    sets_of_training_val_indices = []
    sets_of_test_indices = []

    kf = KFold(n_splits=5, shuffle=True)
    for train_indices, test_indices in kf.split(xs, ys):
        train_train_indices, train_val_indices = train_test_split(train_indices, test_size=0.1)

        sets_of_training_train_indices.append(train_train_indices)
        sets_of_training_val_indices.append(train_val_indices)
        sets_of_test_indices.append(test_indices)

    return ThreewayKFold(sets_of_training_train_indices, sets_of_training_val_indices, sets_of_test_indices)


def create_output_directory(input_file_path, containing_dir_path):
    new_dir_name = f'{input_file_path.stem}_kfold'
    new_output_directory = Path(containing_dir_path, new_dir_name)
    new_output_directory.mkdir(exist_ok=True)

    return new_output_directory


def save_kfold_indices(output_dir, kfold_indices):
    for fold_number in range(len(kfold_indices.train_indices)):
        base = f'fold_{fold_number}'

        pickle_dump(
            out_file=os.path.join(output_dir, f'{base}_train.pkl'),
            item=kfold_indices.train_indices[fold_number]
        )
        pickle_dump(
            out_file=os.path.join(output_dir, f'{base}_val.pkl'),
            item=kfold_indices.val_indices[fold_number]
        )
        pickle_dump(
            out_file=os.path.join(output_dir, f'{base}_test.pkl'),
            item=kfold_indices.test_indices[fold_number]
        )


if __name__ == '__main__':
    main()
