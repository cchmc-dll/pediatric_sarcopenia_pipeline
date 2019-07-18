import os
import cv2

import numpy as np
from nilearn.image import new_img_like
from keras.preprocessing.image import img_to_array

from unet3d.utils.utils import resize, read_image_files
from .utils import crop_img, crop_img_to, read_image

from skimage import io


def find_downsized_info(training_data_files, input_shape):
    foreground = get_complete_foreground(training_data_files)
    crop_slices = crop_img(foreground, return_slices=True, copy=True)
    cropped = crop_img_to(foreground, crop_slices, copy=True)
    final_image = resize(cropped, new_shape=input_shape, interpolation="nearest")
    return crop_slices, final_image.affine, final_image.header


def get_cropping_parameters(in_files):
    if len(in_files) > 1:
        foreground = get_complete_foreground(in_files)
    else:
        foreground = get_foreground_from_set_of_files(in_files[0], return_image=True)
    return crop_img(foreground, return_slices=True, copy=True)


def reslice_image_set(in_files, image_shape, out_files=None, label_indices=None, crop=False):
    if crop:
        crop_slices = get_cropping_parameters([in_files])
    else:
        crop_slices = None
    images = read_image_files(in_files, image_shape=image_shape, crop=crop_slices, label_indices=label_indices)
    if out_files:
        for image, out_file in zip(images, out_files):
            image.to_filename(out_file)
        return [os.path.abspath(out_file) for out_file in out_files]
    else:
        return images


def get_complete_foreground(training_data_files):
    for i, set_of_files in enumerate(training_data_files):
        subject_foreground = get_foreground_from_set_of_files(set_of_files)
        if i == 0:
            foreground = subject_foreground
        else:
            foreground[subject_foreground > 0] = 1

    return new_img_like(read_image(training_data_files[0][-1]), foreground)


def get_foreground_from_set_of_files(set_of_files, background_value=0, tolerance=0.00001, return_image=False):
    for i, image_file in enumerate(set_of_files):
        image = read_image(image_file)
        is_foreground = np.logical_or(image.get_data() < (background_value - tolerance),
                                      image.get_data() > (background_value + tolerance))
        if i == 0:
            foreground = np.zeros(is_foreground.shape, dtype=np.uint8)

        foreground[is_foreground] = 1
    if return_image:
        return new_img_like(image, foreground)
    else:
        return foreground


def normalize_data(data, mean, std):
    data -= mean[:, np.newaxis, np.newaxis, np.newaxis]
    data /= std[:, np.newaxis, np.newaxis, np.newaxis]
    return data


def normalize_data_storage(data_storage):
    means = list()
    stds = list()
    for index in range(data_storage.shape[0]):
        data = data_storage[index]
        means.append(data.mean(axis=(1, 2, 3)))
        stds.append(data.std(axis=(1, 2, 3)))
    mean = np.asarray(means).mean(axis=0)
    std = np.asarray(stds).mean(axis=0)
    for index in range(data_storage.shape[0]):
        data_storage[index] = normalize_data(data_storage[index], mean, std)
    return data_storage


# For 2D images
def normalize_data_2D(data, mean, std):
    data -= mean[:, np.newaxis, np.newaxis]
    data /= std[:, np.newaxis, np.newaxis]
    return data


def normalize_data_storage_2D(data_storage):
    means = list()
    stds = list()
    for index in range(data_storage.shape[0]):
        data = data_storage[index]
        if index == 0:
            cv2.imshow('Un-normalized',data[0])
            cv2.waitKey(1000)

        means.append(data.mean(axis=(1, 2)))
        stds.append(data.std(axis=(1, 2)))
        
    mean = np.asarray(means).mean(axis=0)
    std = np.asarray(stds).mean(axis=0)
    for index in range(data_storage.shape[0]):
        data_storage[index] = normalize_data_2D(data_storage[index], mean, std)
        if index == 0:
            img = data_storage[index]
            cv2.imshow('Normalized',img[0])
            cv2.waitKey(1000)
    
    cv2.destroyAllWindows()
    return data_storage


def normalize_clinical_storage(data_storage):
    mean = np.mean(data_storage,axis=0)
    std = np.std(data_storage,axis=0)
    for index in range(data_storage.shape[0]):
        data_storage[index] -= mean
        data_storage[index] = np.divide(data_storage[index],std,out=np.zeros_like(data_storage[index]),where=std!=0)
    return data_storage


## Routines for processing 2D-TIF images

def resize_pad(im,desired_size,label=False): # Preserves aspect ratio and resizes
    old_size = im.shape[:2] # old_size is in (height, width) format
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    # new_size should be in (width, height) format
    if label:
        im = cv2.resize(im, (new_size[1], new_size[0]),interpolation=cv2.INTER_LINEAR)
        im[im<1] = 0
    else:
        im = cv2.resize(im, (new_size[1], new_size[0]),interpolation=cv2.INTER_CUBIC)

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)
    return new_im


def reslice_image_set_TIF(in_files, image_shape, out_files=None, label_indices=None,slice_number=0, use_middle_slice=False):
    images = list()
    for f, file in enumerate(in_files):
        tiff = io.imread(file, plugin='pil')
        if len(tiff.shape) > 2:
            image = tiff[len(tiff) // 2] if use_middle_slice else tiff[slice_number]
        else:
            tiff = io.imread(file)
            image = tiff

            # Check if LABEL
        label = False
        if f == label_indices:
            label = True
            # Only for Muscle Segmentation
            image[image > 0] = 1
        else:
            image[image < 1800] = 1800
            image[image > 2300] = 2300

        # Resize
        if (image_shape[0] != image.shape[0]) or (image_shape[0] != image.shape[1]):
            desired_size = image_shape[0]
            image = resize_pad(image,desired_size,label)

        print('After Resizing image max: ', np.max(image))
        images.append(image)

    if out_files:
        for image, out_file in zip(images, out_files):
            image.to_filename(out_file)
        return [os.path.abspath(out_file) for out_file in out_files]
    else:
        return images

