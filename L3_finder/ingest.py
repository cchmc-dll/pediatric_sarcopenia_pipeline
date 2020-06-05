import csv
import functools
import os
import subprocess
import sys

import numpy as np
from pathlib import Path

import pydicom
from tqdm import tqdm


dcm2niix_exe = Path(os.getcwd(), 'ext', 'dcm2niix.exe')


class FormatL3DatasetStep:
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
        first_dcm = next(directory.glob(search_pattern))
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


class MissingImage(RuntimeError):
    pass


def load_pixel_data_from_paths(dicom_paths):
    """ Calculates"""
    first_dataset = pydicom.dcmread(str(dicom_paths[0]))
    first_image = first_dataset.pixel_array
    image_dimensions = first_image.shape
    out_array = np.zeros(
        shape=(len(dicom_paths), image_dimensions[0], image_dimensions[1]),
        dtype=first_image.dtype
    )

    # First loaded path not guaranteed first image in series
    index = int(first_dataset.InstanceNumber) - 1
    try:
        out_array[index] = first_image
        datasets = (pydicom.dcmread(str(path)) for path in dicom_paths[1:])
        for dataset in datasets:
            index = int(dataset.InstanceNumber) - 1
            out_array[index] = dataset.pixel_array
    except IndexError:
        out_array = np.resize(out_array, (index + 1, image_dimensions[0], image_dimensions[1]))
    return out_array


def find_images_and_ydata_in_l3_finder_training_format(
        manifest_csv, dataset_path
):
    study_images = list(find_study_images(dataset_path, manifest_csv))
    sagittal_spacings = find_sagittal_image_spacings(study_images, dataset_path)
    names = np.array([image.name for image in study_images], dtype='object')
    ydata = dict(A=find_axial_l3_offsets(study_images))  # One person picked the L3s for this image -> person A

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
    potential_images = (_build_study_image(dataset_path, row) for row in get_image_info_from(manifest_csv))
    return filter(None.__ne__, potential_images)


def _build_study_image(dataset_path, row):
    """
    Uses weird double for loop because Path#glob returns a generator...
    """
    axial_glob_str = "*{subject_id}/**/SE-{axial_series}-*/".format(
        subject_id=row["subject_id"],
        axial_series=row["axial_series"]
    )
    sagittal_glob_str = "*{subject_id}/**/SE-{sagittal_series}-*/".format(
        subject_id=row["subject_id"],
        sagittal_series=row["sagittal_series"]
    )

    for axial_dir in dataset_path.glob(axial_glob_str):
        for sagittal_dir in dataset_path.glob(sagittal_glob_str):
            return StudyImageSet(axial_dir=axial_dir, sagittal_dir=sagittal_dir, **row)


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


def get_spacing(dcm_dataset):
    spacings = [float(spacing) for spacing in dcm_dataset.PixelSpacing]

    """
    Repeats the y spacing for now as we haven't implemented this for frontal scans
    which is where that might make more senese.
    """
    return np.array([spacings[0], spacings[1], spacings[1]], dtype=np.float32)


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


class Subject:
    def __init__(self, path):
        self.path = path

    @property
    def id_(self):
        return self.path.name.split('_')[-1]


KNOWN_ORIENTATIONS = {
    (1, 0, 0, 0, 1, 0): 'axial',
    (-1, 0, 0, 0, -1, 0): 'axial',
    (0, 1, 0, -1, 0, 0): 'axial',
    (0, 1, 0, 0, 0, -1): 'sagittal',
    (-1, 0, 0, 0, 0, -1): 'sagittal',
    (1, 0, 0, 0, 0, -1): 'coronal',
}


def get_orientation(orientation_array):
    rounded_orientation = [round(float(x)) for x in orientation_array]
    return KNOWN_ORIENTATIONS[tuple(rounded_orientation)]


class ImageSeries:
    def __init__(self, subject, series_path, accession_path):
        self.subject = subject
        self.series_path = series_path
        self.accession_path = accession_path

        self._first_dataset = None

    @property
    def pixel_data(self):
        return load_pixel_data_from_paths(
            dicom_paths=list(self.series_path.iterdir())
        )

    @property
    def spacing(self):
        """pseudo spacing with y replaced for the L3 finder"""
        return get_spacing(self._first_dcm_dataset)

    @property
    def true_spacing(self):
        """actual spacing array for axial sma calculation"""
        return [float(spacing) for spacing in self._first_dcm_dataset.PixelSpacing]

    @property
    def resolution(self):
        ds = self._first_dcm_dataset
        return (ds.Rows, ds.Columns)

    @property
    def orientation(self):
        try:
            return get_orientation(self._first_dcm_dataset.ImageOrientationPatient)
        except KeyError as e:
            print(
                "Unknown orientation for for subject: {}. Image Path: {}".format(
                    self.subject.id_,
                    self._first_dcm_path
                ),
                file=sys.stderr
            )
            raise e
        except AttributeError as e:
            print("Attribute error for pt -", self.subject.id_,file=sys.stderr)
            raise e


    @property
    def _first_dcm_dataset(self):
        if not self._first_dataset:
            path = self._first_dcm_path
            # print("accessing", self.subject.id_, file=sys.stderr)
            # print(path, file=sys.stderr)
            self._first_dataset = pydicom.read_file(path, force=True)

        return self._first_dataset

    @property
    def _first_dcm_path(self):
        return str(next(self.series_path.iterdir()).as_posix())

    @property
    def slice_thickness(self):
        return float(self._first_dcm_dataset.SliceThickness)

    def image_at_pos_in_px(self, pos):
        return self.pixel_data[self.image_index_at_pos(pos)]

    def image_index_at_pos(self, pos):
        return int(round(pos / self.slice_thickness))


def find_subjects(dataset_dir):
    for subject_path in Path(dataset_dir).iterdir():
        yield Subject(path=subject_path)


def find_series(subject):
    for accession_path in subject.path.iterdir():
        for series_path in accession_path.iterdir():
            yield ImageSeries(
                subject=subject,
                series_path=series_path,
                accession_path=accession_path
            )


def create_sagittal_mips_from_study_images(study_images):
    # done here so you can require this module w/o loading tensorflow
    from L3_finder.preprocess import create_sagittal_mip
    invalid_images = []
    mips = []
    for image in tqdm(study_images):
        try:
            mips.append(create_sagittal_mip(image.sagittal_pixel_data()))
        except MissingImage as e:
            invalid_images.append(study_images)
            tqdm.write(e)
            invalid_images.append(study_images)

    return mips, invalid_images


def separate_series(series):
    excluded_series = []

    def same_orientation(series, orientation):
        try:
            return series.orientation == orientation
        except AttributeError as e:
            print(
                "Error when determining series orientation for subject:",
                series.subject.id_,
                file=sys.stderr
            )
            excluded_series.append(series)
            return False
    sag_filter = functools.partial(same_orientation, orientation='sagittal')
    axial_filter = functools.partial(same_orientation, orientation='axial')

    sagittal_series = list(filter(sag_filter, series))
    axial_series = list(filter(axial_filter, series))

    return sagittal_series, axial_series, excluded_series
