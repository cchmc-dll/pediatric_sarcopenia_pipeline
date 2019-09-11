from collections import namedtuple

import SimpleITK as sitk
import cv2
import numpy as np
from scipy.ndimage import zoom

from ct_slice_detection.io.preprocessing import preprocess_to_8bit
from ct_slice_detection.utils.testing_utils import preprocess_test_image


def create_mip_from_path(path):
    image_data = load_nifti_data(path)
    sliced_image = slice_middle_images(image_data)
    flipped_image = np.flip(sliced_image)
    return create_mip(flipped_image)


def load_nifti_data(path):
    sitk_img = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(sitk_img)


def slice_middle_images(image_data, offset=6):
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


def normalise_spacing_and_preprocess(dataset, new_spacing=1):
    """Zooms all images to make pixel spacings equal. The other group's paper
    di this, but I'm not sure we really need to... """
    consistently_spaced_images = [

        preprocess_to_8bit(normalize_spacing(image, image_spacing))
        for image, image_spacing
        in zip(dataset['images_s'], dataset['spacings'])
    ]

    new_image_heights = (image.shape[0] for image in consistently_spaced_images)
    return consistently_spaced_images, new_image_heights


def normalize_spacing(image, spacing, desired_spacing=1):
    zoom_factor = [spacing[2] / desired_spacing, spacing[0] / desired_spacing]
    return zoom(image, zoom_factor)


def handle_ydata(ydata):
    """ct-slice-detection application expects ydata to be in this weird
    dict format corresponding to multiple people manually finding L3,
    so this handles that case. Numpy doesn't save dictionaries well, and
    depending on whether you're loading the saved file vs generating it live,
    handles it a little differently"""
    try:
        return ydata.tolist()['A']
    except AttributeError:
        return ydata['A']


def create_sagittal_mip(sagittal_series_data):
    return create_mip(slice_middle_images(sagittal_series_data))


PreprocessedImage = namedtuple(
    'PreprocessedImage', ['pixel_data', 'unpadded_height']
)


def preprocess_images(images, spacings):
    for image, spacing in zip(images, spacings):
        new_image = normalize_spacing(image, spacing)
        new_image = preprocess_to_8bit(new_image)
        height = new_image.shape[0]

        new_image = expand_axes(new_image)
        yield PreprocessedImage(
            pixel_data=new_image,
            unpadded_height=height
        )


def expand_axes(image):
    result = preprocess_test_image(image)
    return result[:, :, np.newaxis]