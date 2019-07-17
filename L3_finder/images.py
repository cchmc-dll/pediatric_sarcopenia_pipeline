import csv
import os
import subprocess
import numpy as np
from collections import namedtuple
from pathlib import Path

import pydicom

from L3_finder.preprocess import create_mip_from_path

dcm2niix_exe = Path(os.getcwd(), 'ext', 'dcm2niix.exe')
# dataset_path = Path("\\\\vnas1\\root1\\Radiology\\SHARED\\Elan\\Projects\\Skeletal Muscle Project\\Dataset2")
# sagittal_csv_path = Path("\\\\vnas1\\root1\\Radiology\\SHARED\\Elan\\Projects\\Skeletal Muscle Project\\sagittal_series_remaining.csv")
# nifti_out_dir = Path.cwd().joinpath('tests', 'data', 'nifti_out')
# CTImage = namedtuple(
#     'DcmImage',
#     [
#         'subject_id',
#         'axial_series',
#         'axial_l3',
#         'sagittal_series',
#         'sagittal_midsag'
#     ]
# )


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


def find_images_and_metadata(manifest_csv, dataset_path, intermediate_nifti_dir):
    study_images = list(find_study_images(dataset_path, manifest_csv))
    sagittal_spacings = find_sagittal_image_spacings(study_images, dataset_path)
    names = np.fromiter((image.name for image in study_images), dtype='S5')
    ydata = dict(A=find_axial_l3_offsets(study_images))  # One person picked the L3s for this image -> person A
    sagittal_mips = create_sagittal_mips(study_images, intermediate_nifti_dir)

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
    potential_images = (build_study_image(dataset_path, row) for row in get_image_info_from(manifest_csv))
    return filter(None.__ne__, potential_images)


def build_study_image(dataset_path, row):
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


def create_sagittal_mips(study_images, nifti_out_dir):
    def convert_to_nifti(image):
        output_path = Path(nifti_out_dir, f'{image.subject_id}.nii')
        if output_path.exists():
            print(f'{output_path.name} already exists, using existing nifti file')
        else:
            nifti_from_dcm_image(image, nifti_out_dir)
        return output_path

    nifti_paths = map(convert_to_nifti, study_images)
    mips = [create_mip_from_path(p) for i, p in enumerate(nifti_paths)]
    return np.array(mips)


def nifti_from_dcm_image(study_image, nifti_out_dir):
    cmd = [str(dcm2niix_exe), '-s', 'y', '-f', study_image.subject_id, '-o', str(nifti_out_dir), str(study_image.sagittal_dir)]
    subprocess.check_call(cmd)


def find_sagittal_image_spacings(study_images, dataset_path):
    datasets = (image.get_dicom_dataset(orientation='sagittal') for image in study_images)

    def get_spacing(dataset):
        spacings = [float(spacing) for spacing in dataset.PixelSpacing]
        return np.array([spacings[0], spacings[1], float(dataset.SliceThickness)], dtype=np.float32)

    spacings = [get_spacing(ds) for ds in datasets]
    return np.array(spacings, dtype=np.float32)


def find_axial_l3_offsets(study_images):
    l3_datasets = (image.get_axial_l3_dataset() for image in study_images)

    def get_offset(dataset):
        return np.float32(dataset.SliceLocation)

    return np.fromiter(map(get_offset, l3_datasets), dtype=np.float32)

# ct_image_generator = (dicom_src_dir_for(row['subject_id'], row['series']) for row in get_image_info_from())
# for ct_image in ct_image_generator:
#     for src_dir in ct_image.src_dirs:
#         cmd = [str(dcm2niix_exe), '-s', 'y', '-f', ct_image.subject_id, '-o', str(nifti_out_dir), str(src_dir)]
#         print(cmd)
#         subprocess.check_call(cmd)

