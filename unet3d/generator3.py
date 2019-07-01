import os
import copy
from random import shuffle
import itertools

import numpy as np

from .utils import pickle_dump, pickle_load
from .utils.patches import compute_patch_indices, get_random_nd_index, get_patch_from_3d_data
from .augment import augment_data, random_permutation_x_y
from pyimagesearch.callbacks.datagenerator3D import DataGenerator_3D
from .utils.generator_utils import get_validation_split, get_number_of_patches, get_number_of_steps   

def get_training_and_validation_generators(data_file, batch_size, n_labels, training_keys_file, validation_keys_file,
                                           data_split=0.8, overwrite=False, labels=None, augment=False,
                                           augment_flip=True, augment_distortion_factor=0.25, patch_shape=None,
                                           validation_patch_overlap=0, training_patch_start_offset=None,
                                           validation_batch_size=None, skip_blank=True, permute=False,reduce=0):
    """
    Creates the training and validation generators that can be used when training the model.
    :param skip_blank: If True, any blank (all-zero) label images/patches will be skipped by the data generator.
    :param validation_batch_size: Batch size for the validation data.
    :param training_patch_start_offset: Tuple of length 3 containing integer values. Training data will randomly be
    offset by a number of pixels between (0, 0, 0) and the given tuple. (default is None)
    :param validation_patch_overlap: Number of pixels/voxels that will be overlapped in the validation data. (requires
    patch_shape to not be None)
    :param patch_shape: Shape of the data to return with the generator. If None, the whole image will be returned.
    (default is None)
    :param augment_flip: if True and augment is True, then the data will be randomly flipped along the x, y and z axis
    :param augment_distortion_factor: if augment is True, this determines the standard deviation from the original
    that the data will be distorted (in a stretching or shrinking fashion). Set to None, False, or 0 to prevent the
    augmentation from distorting the data in this way.
    :param augment: If True, training data will be distorted on the fly so as to avoid over-fitting.
    :param labels: List or tuple containing the ordered label values in the image files. The length of the list or tuple
    should be equal to the n_labels value.
    Example: (10, 25, 50)
    The data generator would then return binary truth arrays representing the labels 10, 25, and 30 in that order.
    :param data_file: hdf5 file to load the data from.
    :param batch_size: Size of the batches that the training generator will provide.
    :param n_labels: Number of binary labels.
    :param training_keys_file: Pickle file where the index locations of the training data will be stored.
    :param validation_keys_file: Pickle file where the index locations of the validation data will be stored.
    :param data_split: How the training and validation data will be split. 0 means all the data will be used for
    validation and none of it will be used for training. 1 means that all the data will be used for training and none
    will be used for validation. Default is 0.8 or 80%.
    :param overwrite: If set to True, previous files will be overwritten. The default mode is false, so that the
    training and validation splits won't be overwritten when rerunning model training.
    :param permute: will randomly permute the data (data must be 3D cube)
    :return: Training data generator, validation data generator, number of training steps, number of validation steps
    """
    if not validation_batch_size:
        validation_batch_size = batch_size

    training_list, validation_list = get_validation_split(data_file,
                                                          data_split=data_split,
                                                          overwrite=overwrite,
                                                          training_file=training_keys_file,
                                                          validation_file=validation_keys_file)
    print(" Pickle Load Complete ")
  
    # Set the number of training and testing samples per epoch correctly
    num_validation_patches,all_patches,validation_list_valid = get_number_of_patches(data_file, validation_list, patch_shape,
                                                                     skip_blank=skip_blank,
                                                                     patch_overlap=validation_patch_overlap)
    num_validation_steps = get_number_of_steps(num_validation_patches,validation_batch_size)
    print("Number of validation steps: ", num_validation_steps)
    
    # Use number of validation patches out of all patches to quickly compute the training patches as an approximation. Saves lot of time.
    # validation_percent = num_validation_patches/all_patches

    #num_training_patches = get_number_of_patches_approx(data_file, training_list, patch_shape,
    #                                                                 skip_blank=skip_blank,
    #                                                                 patch_overlap=validation_patch_overlap,
    #                                                                 validation_percent=validation_percent)
    
    num_training_patches,all_patches,training_list_valid = get_number_of_patches(data_file, training_list, patch_shape,
                                                                     skip_blank=skip_blank,
                                                                     patch_overlap=validation_patch_overlap)
    
    num_training_steps = get_number_of_steps(num_training_patches, batch_size)
    print("Number of training steps: ", num_training_steps)

    training_generator = DataGenerator_3D(data_file, training_list_valid,
                                        batch_size=batch_size,
                                        n_labels=n_labels,
                                        labels=labels,
                                        augment=augment,
                                        augment_flip=augment_flip,
                                        augment_distortion_factor=augment_distortion_factor,
                                        patch_shape=patch_shape,
                                        patch_overlap=0,
                                        patch_start_offset=training_patch_start_offset,
                                        skip_blank=skip_blank,
                                        permute=permute,reduce=reduce)
   
    print(" Training generator ready ")

    validation_generator = DataGenerator_3D(data_file, validation_list_valid,
                                          batch_size=validation_batch_size,
                                          n_labels=n_labels,
                                          labels=labels,
                                          patch_shape=patch_shape,
                                          patch_overlap=validation_patch_overlap,
                                          skip_blank=skip_blank,reduce=reduce)
    print(" Validation generator ready ")
    return training_generator, validation_generator, num_training_steps, num_validation_steps
