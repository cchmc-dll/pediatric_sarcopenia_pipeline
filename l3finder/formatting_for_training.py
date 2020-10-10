import csv
import functools
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pydicom
from tqdm import tqdm

from l3finder.ingest import ImageSeries, load_pixel_data_from_paths, get_spacing

dcm2niix_exe = Path(os.getcwd(), 'ext', 'dcm2niix.exe')


class FormatL3DatasetStep:
    """
    Step that coordinates caching the sagittal images into the format
    used in the L3 finder.
    """
    def __init__(self, cached_file_path, manifest_csv_path, dataset_path):
        self._cached_file_path = cached_file_path
        self._manifest_csv_path = manifest_csv_path
        self._dataset_path = dataset_path

    def load(self):
        if not self._cached_file_path.exists():
            raise FileNotFoundError

        return np.load(str(self._cached_file_path))

    def __call__(self):
        return find_images_and_ydata_in_l3_finder_training_format(self._manifest_csv_path, self._dataset_path)

    def save(self, data_for_l3_finder):
        np.savez_compressed(str(self._cached_file_path), **data_for_l3_finder)


class StudyImageSet:
    def __init__(self, subject_id, axial_series, axial_l3, sagittal_series, sagittal_midsag, sagittal_dir, axial_dir):
        self.subject_id = subject_id
        self.axial_series = axial_series
        self.axial_l3 = axial_l3
        self.sagittal_series = sagittal_series
        self.sagittal_midsag = sagittal_midsag
        self.sagittal_dir = sagittal_dir
        self.axial_dir = axial_dir

    def get_dicom_dataset(self, orientation, search_pattern='*.dcm'):
        directory = getattr(self, '{}_dir'.format(orientation))
        try:
            first_dcm = next(directory.glob(search_pattern))
        except StopIteration:
            pass
        return pydicom.dcmread(str(first_dcm))

    def get_axial_l3_dataset(self):
        pattern = 'IM-CT-{}-*'.format(self.axial_l3)
        return self.get_dicom_dataset(orientation='axial', search_pattern=pattern)

    @property
    def name(self):
        return str(self.subject_id)

    def _pixel_data(self, orientation, search_pattern='*.dcm'):
        directory = getattr(self, '{}_dir'.format(orientation))
        dcm_paths = list(directory.glob(search_pattern))
        try:
            return load_pixel_data_from_paths(dcm_paths)
        except IndexError as e:
            raise MissingImage(
                "Image missing for subject: {subject}, error msg: {err}".format(
                    subject=self.subject_id,
                    err=str(e)
                )
            ) from e

    sagittal_pixel_data = functools.partialmethod(_pixel_data, orientation='sagittal')
    axial_pixel_data = functools.partialmethod(_pixel_data, orientation='axial')

    def to_sagittal_series(self):
        return ImageSeries(
            subject=None,
            series_path=self.sagittal_dir,
            accession_path=self.sagittal_dir
        )


class MissingImage(RuntimeError):
    pass


def find_images_and_ydata_in_l3_finder_training_format(
        manifest_csv, dataset_path
):
    # all_subjects = find_subjects(dataset_path)
    # manifest_subs = set([r["subject_id"] for r in get_image_info_from(manifest_csv)])
    #
    # valid_subs = [sub for sub in all_subjects if sub in manifest_subs]
    #
    # all_series = [sub.find_series() for sub in valid_subs]


    study_images = list(find_study_images(dataset_path, manifest_csv))
    sagittal_spacings = find_sagittal_image_spacings(study_images, dataset_path)
    names = np.array([image.name for image in study_images], dtype='object')
    ydata = dict(A=find_axial_l3_offsets(study_images))  # One person picked the L3s for this preprocessed_image -> person A

    print("Creating sagittal mips...", file=sys.stderr)
    sagittal_mips, invalid_images = create_sagittal_mips_from_study_images(study_images)
    sagittal_mips = np.array(sagittal_mips)

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
    potential_images = [
        _build_study_image(dataset_path, row)
        for row
        in get_image_info_from(manifest_csv)
    ]
    return filter(None.__ne__, potential_images)


def _build_study_image(dataset_path, row):
    """
    Uses weird double for loop because Path#glob returns a generator...
    """
    axial_glob_str = "*_{subject_id}/**/SE-{axial_series}-*/".format(
        subject_id=row["subject_id"],
        axial_series=row["axial_series"]
    )
    sagittal_glob_str = "*_{subject_id}/**/SE-{sagittal_series}-*/".format(
        subject_id=row["subject_id"],
        sagittal_series=row["sagittal_series"]
    )

    try:
        axial_dir = next(dataset_path.glob(axial_glob_str))
    except StopIteration:
        print("No axial for:", row["subject_id"])
        axial_dir = None
        return None

    try:
        sagittal_dir = next(dataset_path.glob(sagittal_glob_str))
    except StopIteration:
        sagittal_dir = None
        print("No sagittal for:", row["subject_id"])
        return None

    return StudyImageSet(axial_dir=axial_dir, sagittal_dir=sagittal_dir, **row)

    # for axial_dir in dataset_path.glob(axial_glob_str):
    #     for sagittal_dir in dataset_path.glob(sagittal_glob_str):
    #         return StudyImageSet(axial_dir=axial_dir, sagittal_dir=sagittal_dir, **row)


def get_image_info_from(csv_path):
    with open(csv_path) as csv_path:
        csv_reader = csv.DictReader(csv_path)
        yield from csv_reader


def nifti_from_dcm_image(study_image, nifti_out_dir):
    cmd = [str(dcm2niix_exe), '-s', 'y', '-f', study_image.subject_id, '-o', str(nifti_out_dir), str(study_image.sagittal_dir)]
    subprocess.check_call(cmd)


def find_sagittal_image_spacings(study_images, dataset_path):
    datasets = (image.get_dicom_dataset(orientation='sagittal') for image in study_images)


    spacings = [get_spacing(ds) for ds in datasets]
    return np.array(spacings, dtype=np.float32)


def find_axial_l3_offsets(study_images):
    l3_datasets = [image.get_axial_l3_dataset() for image in study_images]
    sag_datasets = [image.get_dicom_dataset(orientation='sagittal') for image in study_images]

    offsets_in_px = []
    for l3_ds, sag_ds in zip(l3_datasets, sag_datasets):
        thickness = float(l3_ds.SliceThickness)
        slice_index = int(l3_ds.InstanceNumber) - 1
        y_spacing = float(sag_ds.PixelSpacing[1])

        offsets_in_px.append(slice_index * thickness / y_spacing)

    return np.array(offsets_in_px, dtype=np.float32)


def create_sagittal_mips_from_study_images(study_images):
    # done here so you can require this module w/o loading tensorflow
    from l3finder.preprocess import create_sagittal_mip
    invalid_images = []
    mips = []
    for image in tqdm(study_images):
        try:
            series = image.to_sagittal_series()
            mips.append(create_sagittal_mip(series).pixel_data)
        except MissingImage as e:
            invalid_images.append(study_images)
            tqdm.write(e)
            invalid_images.append(study_images)

    return np.array(mips), invalid_images