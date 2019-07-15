import SimpleITK as sitk
import numpy as np


def create_mip_from_path(path):
    image_data = load_nifti_data(path)
    sliced_image = slice_middle_images(image_data)
    flipped_image = np.flip(sliced_image)
    return create_mip(flipped_image)


def load_nifti_data(path):
    sitk_img = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(sitk_img)


def slice_middle_images(image_data, offset=4):
    x_dim = image_data.shape[0]
    mid_index = x_dim // 2
    return image_data[mid_index - offset:mid_index + offset]


def create_mip(np_img):
    return np.amax(np_img, axis=0)
