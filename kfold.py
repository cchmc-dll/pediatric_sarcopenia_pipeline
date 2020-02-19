import os
from argparse import ArgumentParser, FileType
from collections import namedtuple
from pathlib import Path
import pickle

import numpy as np
import tables
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold, train_test_split

from unet3d.utils import pickle_dump


ThreewayKFold = namedtuple('ThreewayKFold', 'train_indices val_indices test_indices xs ys subject_ids')
TrainTestSplit = namedtuple('TrainTestSplit', 'training_xs_to_kfold test_set training_indices_to_kfold test_indices')


def main():
    args = parse_args()

    with tables.open_file(args.input_file) as data_file:
        xs = np.array(data_file.root.imdata)
        ys = np.array(data_file.root.truth)
        subject_ids = np.array(data_file.root.subject_ids)

    train_test_split = get_training_and_test_split(
        xs=xs,
        existing_test_set_file=args.existing_test_set_file,
    )

    kfold = augment_if_desired(
        kfold=get_kfold(xs, ys, subject_ids, train_test_split),
        should_augment=args.augment,
        samples_per_image=args.samples_per_image,
    )

    if args.augment:
        save_augmented_dataset(input_file_path=args.input_file, kfold=kfold)

    kfold_directory = create_output_directory(
        input_file_path=Path(args.input_file),
        containing_dir_path=Path(args.output_dir),
        did_augment=args.augment
    )

    save_kfold_indices(kfold_directory, kfold)


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('input_file', help='Input .h5 file')
    parser.add_argument('output_dir', help='Directory where dir of kfold files will be created')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--samples_per_image', default=10, type=int, help='Number of samples of each image to take with augmentation')
    parser.add_argument('--existing_test_set_file', type=FileType('rb'), help="Path to .pkl file with existing test set to reuse")

    return parser.parse_args()


def get_training_and_test_split(xs, existing_test_set_file, test_size=0.2):
    indices = np.arange(len(xs))

    if existing_test_set_file:
        existing_test_indices = pickle.load(existing_test_set_file)

        return TrainTestSplit(
            training_xs_to_kfold=np.delete(arr=xs, obj=existing_test_indices, axis=0),
            test_set=xs[existing_test_indices],
            training_indices_to_kfold=np.delete(arr=indices, obj=existing_test_indices),
            test_indices=existing_test_indices
        )
    else:
        return TrainTestSplit(*train_test_split(xs, indices, test_size=0.2))


def get_kfold(xs, ys, subject_ids, train_test_split):
    assert (
        len(xs)
        == len(ys)
        == (len(train_test_split.training_indices_to_kfold) + len(train_test_split.test_indices))
    )

    sets_of_training_train_indices = []
    sets_of_training_val_indices = []
    repeated_set_of_test_indices = []

    # indices = np.arange(len(xs))

    # training_xs_to_kfold, test_set, training_indices_to_kfold, test_indices = train_test_split(
        # xs, indices, test_size=0.2
    # )
    training_ys_to_kfold = ys[train_test_split.training_indices_to_kfold]

    kf = KFold(n_splits=5, shuffle=True)
    for fold_train_indices, fold_val_indices in kf.split(train_test_split.training_xs_to_kfold, training_ys_to_kfold):
        # kf.split gets the indices into the training data array for train and
        # val. We need the indices into the actual dataset, so that's why this
        # selects the values from the training_indices_to_kfold array.
        sets_of_training_train_indices.append(train_test_split.training_indices_to_kfold[fold_train_indices])
        sets_of_training_val_indices.append(train_test_split.training_indices_to_kfold[fold_val_indices])
        # Appends the same test set over and over b/c already written code
        # expects different test sets each time (this was the mistake I made
        # because I didn't know you were supposed to hold a single test set)
        repeated_set_of_test_indices.append(train_test_split.test_indices)

    return ThreewayKFold(
        sets_of_training_train_indices,
        sets_of_training_val_indices,
        repeated_set_of_test_indices,
        xs,
        ys,
        subject_ids
    )


def augment_if_desired(kfold, should_augment=False, **augmentation_options):
    if should_augment:
        return augment_data(kfold, **augmentation_options)
    else:
        return kfold


def augment_data(kfold_indices, samples_per_image):
    new_xs = []
    new_ys = []
    new_subject_ids=[]

    index = len(kfold_indices.xs)  # appending new images to end of current list
    new_train_indices = []
    for fold_number in range(len(kfold_indices.train_indices)):
        print('augmenting fold: ', fold_number)
        subset_of_xs = kfold_indices.xs[kfold_indices.train_indices[fold_number]]
        subset_of_ys = kfold_indices.ys[kfold_indices.train_indices[fold_number]]

        x_samples, y_samples = get_augmented_samples(subset_of_xs, subset_of_ys, samples_per_image)

        current_fold_train_indices = []

        new_xs.extend(x_samples)
        new_ys.extend(y_samples)

        for _ in range(len(x_samples)):
            current_fold_train_indices.append(index)
            new_subject_ids.append(f'a{index}')
            index = index + 1

        new_train_indices.append(current_fold_train_indices)

    return ThreewayKFold(
        train_indices=new_train_indices,
        val_indices=kfold_indices.val_indices,
        test_indices=kfold_indices.test_indices,
        xs=np.concatenate([kfold_indices.xs, new_xs]),
        ys=np.concatenate([kfold_indices.ys, new_ys]),
        subject_ids=np.concatenate([kfold_indices.subject_ids, new_subject_ids])
    )


def get_augmented_samples(subset_of_xs, subset_of_ys, samples_per_image):
    # pipeline = get_augmentation_pipeline(subset_of_xs, subset_of_ys)
    # # samples = pipeline.sample(len(subset_of_xs) * samples_per_image)
    # samples = pipeline.sample(10)
    # return samples
    datagen_args = dict(
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.5,
        rotation_range=35,
        shear_range=15,
        width_shift_range=0.3,
        height_shift_range=0.3,
    )

    seed = 1
    image_datagen = ImageDataGenerator(**datagen_args)
    mask_datagen = ImageDataGenerator(**datagen_args)

    x_samples = []
    y_samples = []

    image_datagen.fit(subset_of_xs, augment=True, seed=seed)
    mask_datagen.fit(subset_of_ys, augment=True, seed=seed)

    image_generator = image_datagen.flow(subset_of_xs, seed=seed)
    mask_generator = image_datagen.flow(subset_of_ys, seed=seed)

    for batch_number, (x_batch, y_batch) in enumerate(zip(image_generator, mask_generator)):
        print('Batch: ', batch_number)

        x_samples.extend(x_batch)
        y_samples.extend(y_batch)

        if len(x_samples) > len(subset_of_xs) * 15:
            break

    return np.array(x_samples), make_mask_boolean(np.array(y_samples))


def make_mask_boolean(mask):
    mask[mask > 0] = 1
    return mask


def save_augmented_dataset(input_file_path, kfold):
    with tables.open_file(_get_augmented_file_path(input_file_path), 'w') as hd5:
        hd5.create_array(hd5.root, 'imdata', kfold.xs)
        hd5.create_array(hd5.root, 'truth', kfold.ys)
        hd5.create_array(hd5.root, 'subject_ids', kfold.subject_ids)


def _get_augmented_file_path(input_file_path):
    old_path = Path(input_file_path)
    return str(Path(old_path.parent, f'{old_path.stem}_aug.h5'))


def create_output_directory(input_file_path, containing_dir_path, did_augment):
    new_dir_name = f'{input_file_path.stem}_kfold_aug' if did_augment else f'{input_file_path.stem}_kfold'
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
