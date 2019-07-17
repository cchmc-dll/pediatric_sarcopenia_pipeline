import SimpleITK as sitk
import cv2
import numpy as np


def create_mip_from_path(path):
    image_data = load_nifti_data(path)
    sliced_image = slice_middle_images(image_data)
    flipped_image = np.flip(sliced_image)
    return create_mip(flipped_image)


def load_nifti_data(path):
    sitk_img = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(sitk_img)


def slice_middle_images(image_data, offset=4):
    x_dim = image_data.shape[0]
    mid_index = x_dim // 2
    return image_data[mid_index - offset:mid_index + offset]


def create_mip(np_img):
    return np.amax(np_img, axis=0)


def threshold(image_data, low, high):
    r = np.copy(image_data)
    r[(r < low) | (r > high)] = np.iinfo(np.int16).min
    return r


def normalize_to_8bit(image_data):
    r = np.full_like(image_data, 1, dtype=np.int8)
    info = np.iinfo(np.int8)
    return cv2.normalize(image_data, None, info.min, info.max, cv2.NORM_MINMAX)


def resize_for_nn(image_data):
    # scale_factor = 384 / image_data.shape[0]
    # target = (image_data.shape[0] * scale_factor, image_data.shape[1] * scale_factor)
    assert image_data.shape[0] == image_data.shape[1], ' Should be square image'
    scaled = cv2.resize(image_data, (384, 384), interpolation=cv2.INTER_AREA)
    trim_amount = (384 - 256) // 2
    return scaled[:, trim_amount:-trim_amount]


def process_image(path):
    mip = create_mip_from_path(str(path))
    mip = threshold(mip, low=100, high=1500)
    norm = normalize_to_8bit(mip)
    print(path, norm.shape)
    return resize_for_nn(norm)