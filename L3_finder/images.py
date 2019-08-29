import csv
import os
import subprocess
import sys

import numpy as np
from collections import namedtuple
from pathlib import Path
from tqdm import tqdm

import pydicom

from L3_finder.preprocess import create_mip_from_path, create_mip, slice_middle_images
from util.pipelines import build_callable_that_loads_from_cache_or_runs_func, CachablePipelineStep

dcm2niix_exe = Path(os.getcwd(), 'ext', 'dcm2niix.exe')


class LoadL3DatasetCachableStep:
    def __init__(self, cached_file_path, manifest_csv_path, dataset_path):
        self._cached_file_path = cached_file_path
        self._manifest_csv_path = manifest_csv_path
        self._dataset_path = dataset_path

    def load(self):
        if not self._cached_file_path.exists():
            raise FileNotFoundError

        return np.load(str(self._cached_file_path))

    def __call__(self):
        return find_images_and_ydata_in_l3_finder_format(self._manifest_csv_path, self._dataset_path)

    def save(self, data_for_l3_finder):
        np.savez_compressed(str(self._cached_file_path), **data_for_l3_finder)


class StudyImage:
    def __init__(self, subject_id, axial_series, axial_l3, sagittal_series, sagittal_midsag, sagittal_dir, axial_dir):
        self.subject_id = subject_id
        self.axial_series = axial_series
        self.axial_l3 = axial_l3
        self.sagittal_series = sagittal_series
        self.sagittal_midsag = sagittal_midsag
        self.sagittal_dir = sagittal_dir
        self.axial_dir = axial_dir

    def get_dicom_dataset(self, orientation, search_pattern='*.dcm'):
        directory = getattr(self, f'{orientation}_dir')
        first_dcm = next(directory.glob(search_pattern))
        return pydicom.dcmread(str(first_dcm))

    def get_axial_l3_dataset(self):
        pattern = f'IM-CT-{self.axial_l3}-*'
        return self.get_dicom_dataset(orientation='axial', search_pattern=pattern)

    @property
    def name(self):
        return str(self.subject_id)

    def pixel_data(self, orientation, search_pattern='*.dcm'):
        directory = getattr(self, f'{orientation}_dir')
        dcm_paths = list(directory.glob(search_pattern))
        first_image = pydicom.dcmread(str(dcm_paths[0])).pixel_array
        image_dimensions = first_image.shape
        out_array = np.ndarray(
            shape=(len(dcm_paths), image_dimensions[0], image_dimensions[1]),
            dtype=first_image.dtype
        )

        out_array[0] = first_image
        pixel_arrays = (pydicom.dcmread(str(path)).pixel_array for path in dcm_paths[1:])
        for index, pixel_array in enumerate(pixel_arrays):
            out_array[index] = pixel_array

        return out_array


def find_images_and_ydata_in_l3_finder_format(manifest_csv, dataset_path):
    study_images = list(find_study_images(dataset_path, manifest_csv))
    sagittal_spacings = find_sagittal_image_spacings(study_images, dataset_path)
    names = np.array([image.name for image in study_images], dtype='object')
    ydata = dict(A=find_axial_l3_offsets(study_images))  # One person picked the L3s for this image -> person A

    print("Creating sagittal mips...", file=sys.stderr)
    sagittal_mips = create_sagittal_mips(study_images)

    assert len(study_images) == len(sagittal_spacings) == len(names) == len(sagittal_mips)

    return dict(
        images_f=sagittal_mips,  # for now...
        images_s=sagittal_mips,
        spacings=sagittal_spacings,
        names=names,
        ydata=ydata,
        num_images=len(sagittal_mips)
    )


def find_study_images(dataset_path, manifest_csv):
    """Potential because folders may not exist..."""
    potential_images = (_build_study_image(dataset_path, row) for row in get_image_info_from(manifest_csv))
    return filter(None.__ne__, potential_images)


def _build_study_image(dataset_path, row):
    """
    Uses weird double for loop because Path#glob returns a generator...
    """
    for axial_dir in dataset_path.glob(f"*{row['subject_id']}/**/SE-{row['axial_series']}-*/"):
        for sagittal_dir in dataset_path.glob(f"*{row['subject_id']}/**/SE-{row['sagittal_series']}-*/"):
            return StudyImage(axial_dir=axial_dir, sagittal_dir=sagittal_dir, **row)


def get_image_info_from(csv_path):
    with open(csv_path) as csv_path:
        csv_reader = csv.DictReader(csv_path)
        yield from csv_reader


def create_sagittal_mips(study_images):
    # def convert_to_nifti(image):
    #     output_path = Path(nifti_out_dir, f'{image.subject_id}.nii')
    #     if output_path.exists():
    #         print(f'{output_path.name} already exists, using existing nifti file')
    #     else:
    #         nifti_from_dcm_image(image, nifti_out_dir)
    #     return output_path

    # nifti_paths = map(convert_to_nifti, study_images)
    # mips = [create_mip_from_path(p) for i, p in enumerate(nifti_paths)]
    # return np.array(mips)
    mips = [
        create_mip(slice_middle_images(image.pixel_data(orientation='sagittal')))
        for image
        in tqdm(study_images)
    ]
    return np.array(mips)

def nifti_from_dcm_image(study_image, nifti_out_dir):
    cmd = [str(dcm2niix_exe), '-s', 'y', '-f', study_image.subject_id, '-o', str(nifti_out_dir), str(study_image.sagittal_dir)]
    subprocess.check_call(cmd)


def find_sagittal_image_spacings(study_images, dataset_path):
    datasets = (image.get_dicom_dataset(orientation='sagittal') for image in study_images)

    def get_spacing(dataset):
        spacings = [float(spacing) for spacing in dataset.PixelSpacing]

        """
        Repeats the y spacing for now as we haven't implemented this for frontal scans
        which is where that might make more senese.
        """
        return np.array([spacings[0], spacings[1], spacings[1]], dtype=np.float32)
        # return np.array([spacings[0], spacings[1], float(dataset.SliceThickness)], dtype=np.float32)

    spacings = [get_spacing(ds) for ds in datasets]
    return np.array(spacings, dtype=np.float32)


def find_axial_l3_offsets(study_images):
    l3_datasets = (image.get_axial_l3_dataset() for image in study_images)
    sag_datasets = (image.get_dicom_dataset(orientation='sagittal') for image in study_images)

    offsets_in_px = []
    for l3_ds, sag_ds in zip(l3_datasets, sag_datasets):
        thickness = float(l3_ds.SliceThickness)
        slice_index = int(l3_ds.InstanceNumber) - 1
        y_spacing = float(sag_ds.PixelSpacing[1])

        offsets_in_px.append(slice_index * thickness / y_spacing)

    return np.array(offsets_in_px, dtype=np.float32)

