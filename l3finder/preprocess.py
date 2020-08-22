from matplotlib import pyplot as plt

from collections import namedtuple
import multiprocessing
import os
import pickle
import traceback

import attr
import SimpleITK as sitk
import cv2
import numpy as np
from scipy.ndimage import zoom
import toolz
from tqdm import tqdm

from ct_slice_detection.io.preprocessing import preprocess_to_8bit
from ct_slice_detection.utils.testing_utils import preprocess_test_image
from util.pipelines import CachablePipelineStep, build_callable_that_loads_from_cache_or_runs_step


def create_mip_from_path(path):
    image_data = load_nifti_data(path)
    sliced_image = slice_middle_images(image_data)
    flipped_image = np.flip(sliced_image)
    return create_mip(flipped_image)


def load_nifti_data(path):
    sitk_img = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(sitk_img)


def slice_middle_images(sagittal_series, mip_thickness_mm=36):
    x_dim = sagittal_series.pixel_data.shape[0]
    mid_index = x_dim // 2

    num_slices = mip_thickness_mm / sagittal_series.slice_thickness
    offset = round(num_slices / 2)
    return sagittal_series.pixel_data[mid_index - offset:mid_index + offset]


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
    assert image_data.shape[0] == image_data.shape[1], ' Should be square preprocessed_image'
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


def create_sagittal_mips_from_series(many_series, cache_dir="", cache=True):
    sagittal_mip_creator = build_callable_that_loads_from_cache_or_runs_step(
        pipeline_step=CreateSagittalMIPsStep(
            cache_dir,
            expected_count=len(many_series),
        ),
        use_cache=cache,
    )

    return sagittal_mip_creator(many_series)


class CreateSagittalMIPsStep(CachablePipelineStep):
    def __init__(self, cache_dir, expected_count):
        self._cache_dir = cache_dir
        self._cache_file_name = "_sagittal_mips.pkl"
        self.expected_count = expected_count

    def load(self):
        with open(self._cache_pickle_path, "rb") as f:
            print("Loading Sagittal MIPs from the cache at:", self._cache_pickle_path)
            try:
                cached_series = pickle.load(f)
            except ImportError:
                print("Import error when loading mips pickle from cache, Recalculating.")
                raise FileNotFoundError

            return cached_series

            # if len(cached_series) == self.expected_count:
                # return cached_series
            # else:
                # print("Cached length different than expected. Recalculating.")
                # raise FileNotFoundError

    @property
    def _cache_pickle_path(self):
        return os.path.join(self._cache_dir, self._cache_file_name)

    def __call__(self, many_series):
        return _create_sagittal_mips_from_series(many_series)

    def save(self, sagittal_mips):
        with open(self._cache_pickle_path, "wb") as f:
            return pickle.dump(sagittal_mips, f)


def _create_sagittal_mips_from_series(many_series):
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        # Could use map, but imap lets me get a progress bar
        mips = list(
            tqdm(
                pool.imap(_load_image_and_create_mip, many_series),
                total=len(many_series),
            )
        )
        pool.close()
        pool.join()

    return [mip for mip in mips if mip is not None]


def _load_image_and_create_mip(one_series):
    return create_sagittal_mip(one_series)

@attr.s(frozen=True)
class MIP:
    pixel_data = attr.ib()
    source_series = attr.ib()

def create_sagittal_mip(one_sagittal_series):
    try:
        pixel_data = create_mip(slice_middle_images(one_sagittal_series))
        one_sagittal_series.free_pixel_data()
        return MIP(pixel_data=pixel_data, source_series=one_sagittal_series)
    except RuntimeError as e:
        print("RuntimeError encountered when creating sagittal mip")
        print(traceback.format_exc())
        return None


@attr.s
class PreprocessedImage:
    pixel_data = attr.ib()
    unpadded_height = attr.ib()
    source_mip = attr.ib()

    @property
    def source_series(self):
        return self.source_mip.source_series




def preprocess_images(mips):
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        images = list(tqdm(pool.imap(_preprocess_mip, mips)))
        pool.close()
        pool.join()

    # images = map(_preprocess_mip, mips)
    return images

def _preprocess_mip(mip):
    new_image = normalize_spacing(
        image=mip.pixel_data,
        spacing=mip.source_series.spacing,
    )

    new_image = preprocess_to_8bit(new_image)
    height = new_image.shape[0]

    new_image = expand_axes(new_image)
    return PreprocessedImage(
        pixel_data=new_image,
        unpadded_height=height,
        source_mip=mip
    )


def expand_axes(image):
    result = preprocess_test_image(image)
    return result[:, :, np.newaxis]


def group_mips_by_dimension(mips):
    def dimension_from_sagittal_mip(sag_mip):
        return sag_mip.preprocessed_image.pixel_data.shape
    return toolz.groupby(dimension_from_sagittal_mip, mips)


