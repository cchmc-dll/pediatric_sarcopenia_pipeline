import os
import copy
from random import shuffle
import itertools
import cv2
import numpy as np

from unet3d.utils.utils import pickle_dump, pickle_load
from unet3d.utils.patches import compute_patch_indices, get_random_nd_index, get_patch_from_3d_data
from unet3d.augment import augment_data, random_permutation_x_y


def get_number_of_steps(n_samples, batch_size):
    if n_samples <= batch_size:
        return n_samples
    elif np.remainder(n_samples, batch_size) == 0:
        return n_samples//batch_size
    else:
        return n_samples//batch_size + 1


def get_validation_split(data_file, training_file, validation_file, data_split=0.8, overwrite=False):
    """
    Splits the data into the training and validation indices list.
    :param data_file: pytables hdf5 data file
    :param training_file:
    :param validation_file:
    :param data_split:
    :param overwrite:
    :return:
    """
    if overwrite or not os.path.exists(training_file):
        print("Creating validation split...")
        nb_samples = data_file.root.imdata.shape[0]
        sample_list = list(range(nb_samples))
        training_list, validation_list = split_list(sample_list, split=data_split)
        pickle_dump(training_list, training_file)
        pickle_dump(validation_list, validation_file)
        return training_list, validation_list
    else:
        print("Loading previous validation split...")
        return pickle_load(training_file), pickle_load(validation_file)


def split_list(input_list, split=0.8, shuffle_list=True):
    if shuffle_list:
        shuffle(input_list)
    n_training = int(len(input_list) * split)
    training = input_list[:n_training]
    testing = input_list[n_training:]
    return training, testing

def get_number_of_patches(data_file, index_list, patch_shape=None, patch_overlap=0, patch_start_offset=None,
                          skip_blank=True):
    if patch_shape:
        valid_patches = list()
        print("Number of images: ", len(index_list))
        index_list = create_patch_index_list(index_list, data_file.root.imdata.shape[-3:], patch_shape, patch_overlap,
                                             patch_start_offset)
        print("Number of patches: ", len(index_list))
        count = 0
        for index in index_list:
            x_list = list()
            y_list = list()
            add_data(x_list, y_list, data_file, index, skip_blank=skip_blank, patch_shape=patch_shape)
            if len(x_list) > 0:
                valid_patches.append(index)
                count += 1
        print("Number of count: ", count)
        return count, len(index_list),valid_patches
    else:
        return len(index_list), len(index_list),index_list

def get_number_of_patches_approx(data_file, index_list, patch_shape=None, patch_overlap=0, patch_start_offset=None,
                          skip_blank=True, validation_percent = 0.25):
    if patch_shape:
        print("Number of images: ", len(index_list))
        index_list = create_patch_index_list(index_list, data_file.root.imdata.shape[-3:], patch_shape, patch_overlap,
                                             patch_start_offset)
        print("Number of patches: ", len(index_list))
        count = round(validation_percent*len(index_list))
        print("Number of count: ", count)
        return count  
    else:
        return len(index_list)

def create_patch_index_list(index_list, image_shape, patch_shape, patch_overlap, patch_start_offset=None):
    patch_index = list()
    for index in index_list:
        if patch_start_offset is not None:
            random_start_offset = np.negative(get_random_nd_index(patch_start_offset))
            patches = compute_patch_indices(image_shape, patch_shape, overlap=patch_overlap, start=random_start_offset)
        else:
            patches = compute_patch_indices(image_shape, patch_shape, overlap=patch_overlap)
        patch_index.extend(itertools.product([index], patches))
    return patch_index

def add_data_segment(x_list, y_list, data_file, index, augment=False, augment_flip=False, augment_distortion_factor=0.25,
             patch_shape=None, skip_blank=True, permute=False, reduce = 0):
    """
    Adds data from the data file to the given lists of feature and target data
    :param skip_blank: Data will not be added if the truth vector is all zeros (default is True).
    :param patch_shape: Shape of the patch to add to the data lists. If None, the whole image will be added.
    :param x_list: list of data to which data from the data_file will be appended.
    :param y_list: list of data to which the target data from the data_file will be appended.
    :param data_file: hdf5 data file.
    :param index: index of the data file from which to extract the data.
    :param augment: if True, data will be augmented according to the other augmentation parameters (augment_flip and
    augment_distortion_factor)
    :param augment_flip: if True and augment is True, then the data will be randomly flipped along the x, y and z axis
    :param augment_distortion_factor: if augment is True, this determines the standard deviation from the original
    that the data will be distorted (in a stretching or shrinking fashion). Set to None, False, or 0 to prevent the
    augmentation from distorting the data in this way.
    :param permute: will randomly permute the data (data must be 3D cube)
    :param reduce: will reduce the image stack by symmetrically by the specified number: i.e if reduce = 2, 1st and last slice will be removed before training.
    :return:
    """  
    data, truth = get_data_from_file_segment(data_file, index, patch_shape=patch_shape)
    #print("\n Adding Data...|index = ", index,' :, Data shape: ', data.shape,'\n')
    if reduce:
        cut = int(reduce/2)
        data = data[:,:,:,cut:data.shape[3]-cut]
        truth = truth[:,:,cut:truth.shape[2]-cut]

    if augment:
        if patch_shape is not None:
            affine = data_file.root.affine[index[0]]
        else:
            affine = data_file.root.affine[index]
        data, truth = augment_data(data, truth, affine, flip=augment_flip, scale_deviation=augment_distortion_factor)

    if permute:
        if data.shape[-3] != data.shape[-2] or data.shape[-2] != data.shape[-1]:
            raise ValueError("To utilize permutations, data array must be in 3D cube shape with all dimensions having "
                             "the same length.")
        data, truth = random_permutation_x_y(data, truth[np.newaxis])
    else:
        truth = truth[np.newaxis]

    if not skip_blank or np.any(truth != 0):
        x_list.append(data)
        y_list.append(truth)

def add_data_classify(x_list, data_file, index, augment=False, augment_flip=False, augment_distortion_factor=0.25,
             patch_shape=None, skip_blank=False, permute=False, reduce = 0):
    """
    Adds data from the data file to the given lists of feature and target data
    :param skip_blank: Data will not be added if the truth vector is all zeros (default is True).
    :param patch_shape: Shape of the patch to add to the data lists. If None, the whole image will be added.
    :param x_list: list of data to which data from the data_file will be appended.
    :param y_list: list of data to which the target data from the data_file will be appended.
    :param data_file: hdf5 data file.
    :param index: index of the data file from which to extract the data.
    :param augment: if True, data will be augmented according to the other augmentation parameters (augment_flip and
    augment_distortion_factor)
    :param augment_flip: if True and augment is True, then the data will be randomly flipped along the x, y and z axis
    :param augment_distortion_factor: if augment is True, this determines the standard deviation from the original
    that the data will be distorted (in a stretching or shrinking fashion). Set to None, False, or 0 to prevent the
    augmentation from distorting the data in this way.
    :param permute: will randomly permute the data (data must be 3D cube)
    :param reduce: will reduce the image stack by symmetrically by the specified number: i.e if reduce = 2, 1st and last slice will be removed before training.
    :return:
    """
    
    data  = data_file.root.imdata[index]
    #print("\n Adding Data...|index = ", index,' :, Data shape: ', data.shape,'\n')
    if reduce:
        cut = int(reduce/2)
        data = data[:,:,:,cut:data.shape[3]-cut]

    if augment:
        if patch_shape is not None:
            affine = data_file.root.affine[index[0]]
        else:
            affine = data_file.root.affine[index]
        data = augment_data(data,None, affine, flip=augment_flip, scale_deviation=augment_distortion_factor)

    if permute:
        if data.shape[-3] != data.shape[-2] or data.shape[-2] != data.shape[-1]:
            raise ValueError("To utilize permutations, data array must be in 3D cube shape with all dimensions having "
                             "the same length.")
        data = random_permutation_x_y(data,None)

    x_list.append(data)

def convert_data(x_list, y_list, n_labels=1, labels=None):
    x = np.asarray(x_list)
    y = np.asarray(y_list)
    if n_labels == 1:
        y[y > 0] = 1
    elif n_labels > 1:
        y = get_multi_class_labels(y, n_labels=n_labels, labels=labels)
    
    #print('x_list len:', len(x_list), '\n')
    #if(x.shape[1] != 256 or y.shape[1] != 256):
    #    print("\n Convert Data... X shape: ", x.shape,' Y shape: ', y.shape, '\n')

    return x, y


def get_multi_class_labels(data, n_labels, labels=None):
    """
    Translates a label map into a set of binary labels.
    :param data: numpy array containing the label map with shape: (n_samples, 1, ...).
    :param n_labels: number of labels.
    :param labels: integer values of the labels.
    :return: binary numpy array of shape: (n_samples, n_labels, ...)
    """
    new_shape = [data.shape[0], n_labels] + list(data.shape[2:])
    y = np.zeros(new_shape, np.int8)
    for label_index in range(n_labels):
        if labels is not None:
            y[:, label_index][data[:, 0] == labels[label_index]] = 1
        else:
            y[:, label_index][data[:, 0] == (label_index + 1)] = 1
    return y


def get_data_from_file_2Dsegment(data_file, index):
    x, y = data_file.root.imdata[index], data_file.root.truth[index]
    return x, y

def add_data_segment_2D(x_list, y_list, data_file, index, 
             patch_shape=None):
    """
    Adds data from the data file to the given lists of feature and target data
    :param skip_blank: Data will not be added if the truth vector is all zeros (default is True).
    :param patch_shape: Shape of the patch to add to the data lists. If None, the whole image will be added.
    :param x_list: list of data to which data from the data_file will be appended.
    :param y_list: list of data to which the target data from the data_file will be appended.
    :param data_file: hdf5 data file.
    :param index: index of the data file from which to extract the data.
    :param augment: if True, data will be augmented according to the other augmentation parameters (augment_flip and
    augment_distortion_factor)
    :param augment_flip: if True and augment is True, then the data will be randomly flipped along the x, y and z axis
    :param augment_distortion_factor: if augment is True, this determines the standard deviation from the original
    that the data will be distorted (in a stretching or shrinking fashion). Set to None, False, or 0 to prevent the
    augmentation from distorting the data in this way.
    :param permute: will randomly permute the data (data must be 3D cube)
    :param reduce: will reduce the image stack by symmetrically by the specified number: i.e if reduce = 2, 1st and last slice will be removed before training.
    :return:
    """  
    data, truth = get_data_from_file_2Dsegment(data_file, index)
    x_list.append(data)
    y_list.append(truth)