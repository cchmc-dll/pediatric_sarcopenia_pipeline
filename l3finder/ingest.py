from matplotlib import pyplot as plt

import csv
import functools
import multiprocessing
import os
import subprocess
import sys
import warnings
from pathlib import Path
import pickle

import attr
import numpy as np
import pydicom
from tqdm import tqdm

from util.reify import reify


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
    datasets = list(
        sorted(
            (pydicom.dcmread(str(p)) for p in dicom_paths),
            key=lambda ds: int(ds.InstanceNumber)
        )
    )

    first_dataset = datasets[0]
    first_image = first_dataset.pixel_array
    image_dimensions = first_image.shape

    out_array = np.zeros(
        shape=(len(datasets), image_dimensions[0], image_dimensions[1]),
        dtype=first_image.dtype
    )

    for index, dataset in enumerate(datasets):
        out_array[index] = dataset.pixel_array

    return out_array



def find_images_and_ydata_in_l3_finder_training_format(
        manifest_csv, dataset_path
):
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


class UnknownOrientation(Exception):
    """
    Encountered an orientation that is not defined in the KNOWN_ORIENTATIONS
    variable.
    """
    def __init__(self, series, msg=None):
        super(UnknownOrientation, self).__init__(msg)
        self.series = series


@attr.s
class ImageSeries:
    position_key = (0x0020, 0x0032)
    ax_z_pos_key = 2
    sag_z_pos_key = 0
    _pixel_data = None
    _first_dataset = None

    subject = attr.ib()
    series_path = attr.ib()
    accession_path = attr.ib()

    @reify
    def id_(self):
        return "{subject_id}-{series_name}".format(
            subject_id=self.subject.id_,
            series_name=self.series_path.name
        )

    @property
    def series_name(self):
        return self.series_path.name

    @property
    def pixel_data(self):
        if self._pixel_data is None:
            self._pixel_data = load_pixel_data_from_paths(
                dicom_paths=list(self.series_path.iterdir())
            )
        return self._pixel_data

    def free_pixel_data(self):
        """Use to free memory if too much pixel_data"""
        self._pixel_data = None

    @reify
    def spacing(self):
        """pseudo spacing with y replaced for the L3 finder"""
        return get_spacing(self._any_dcm_dataset)

    @reify
    def true_spacing(self):
        """actual spacing array for axial sma calculation"""
        return [float(spacing) for spacing in self._any_dcm_dataset.PixelSpacing]

    @reify
    def resolution(self):
        ds = self._any_dcm_dataset
        return (ds.Rows, ds.Columns)

    @reify
    def orientation(self):
        try:
            return get_orientation(self._any_dcm_dataset.ImageOrientationPatient)
        except (KeyError, AttributeError) as e:
            raise UnknownOrientation(series=self) from e

    @reify
    def _any_dcm_dataset(self):
        return pydicom.read_file(self._any_dcm_path, force=True)

    @reify
    def _any_dcm_path(self):
        return str(next(self.series_path.iterdir()).as_posix())

    @reify
    def slice_thickness(self):
        return float(self._any_dcm_dataset.SliceThickness)

    def image_at_pos_in_px(self, pos, sagittal_start_z_pos):
        l3_axial_index, metadata = self.image_index_at_pos(pos, sagittal_start_z_pos)

        if self._pixel_data is None:
            return pydicom.read_file(metadata.l3_axial_image_dcm_path).pixel_array
        else:
            return self.pixel_data[l3_axial_index]

    # Need to undo the spacing normalization, which is done using sagittal spacing[2]
    def image_index_at_pos(self, pos_with_1mm_spacing, sagittal_start_z_pos):
        """1mm spacing is the default coming out of the preprocessing"""
        dataset_path_pairs = self.dataset_path_pairs_in_order

        length = len(dataset_path_pairs)
        series_z_positions = np.empty(
            shape=len(dataset_path_pairs), dtype=np.float32
        )
        dicom_paths = np.empty(shape=length, dtype="object")
        for index, (dataset, path) in enumerate(dataset_path_pairs):
            series_z_positions[index] = dataset[self.position_key][self.ax_z_pos_key]
            dicom_paths[index] = path

        direction = np.sign(series_z_positions[-1] - series_z_positions[0])
        z_position = sagittal_start_z_pos + pos_with_1mm_spacing*direction

        # Finds the closest slice to calculated z_position
        l3_axial_image_index = np.argmin(np.abs(series_z_positions - z_position))

        metadata = L3AxialSliceMetadata(
            sagittal_start_z_pos=sagittal_start_z_pos,
            predicted_z_position=z_position,
            first_axial_pos=series_z_positions[0],
            last_axial_pos=series_z_positions[-1],
            l3_axial_image_index=l3_axial_image_index,
            axial_image_count=len(series_z_positions),
            l3_axial_image_dcm_path=dicom_paths[l3_axial_image_index],
        )

        return l3_axial_image_index, metadata

    @reify
    def number_of_dicoms(self):
        return len(list(self.series_path.iterdir()))

    @property
    def starting_z_pos(self):
        return self.z_range_pair[0]

    @reify
    def z_range_pair(self):
        first_dataset = self.dataset_path_pairs_in_order[0][0]
        last_dataset = self.dataset_path_pairs_in_order[-1][0]

        if self.orientation == 'axial':
            return (
                np.float(first_dataset[self.position_key][self.ax_z_pos_key]),
                np.float(last_dataset[self.position_key][self.ax_z_pos_key]),
            )
        elif self.orientation == 'sagittal':
            direction = -1 if first_dataset[self.position_key][0] > last_dataset[self.position_key][0] else 1
            distance = first_dataset.PixelSpacing[0] * first_dataset.Rows
            first_pos = first_dataset[self.position_key][self.ax_z_pos_key]
            return (
                np.float(first_pos),
                np.float(first_pos + (direction * distance))
            )
        else:
            raise "z_range_pair requested for not supported orientation {}".format(self.orientation)


    @property
    def dataset_path_pairs_in_order(self):
        return sorted(
            ((pydicom.dcmread(str(p)), p) for p in self.series_path.iterdir()),
            key=lambda ds_path_pair: int(ds_path_pair[0].InstanceNumber)
        )


@attr.s
class ConstructedImageSeries:
    axial_series = attr.ib()
    _pixel_data = attr.ib(default=None)

    @property
    def series_name(self):
        return "recon from: " + self.axial_series.series_name

    @property
    def subject(self):
        return self.axial_series.subject

    @property
    def pixel_data(self):
        if self._pixel_data is None:
            self._pixel_data = _construct_sagittal_from_axial_image(
                self.axial_series.pixel_data
            )
        return self._pixel_data

    def free_pixel_data(self):
        """Use to free memory if too much pixel_data"""
        self._pixel_data = None
        self.axial_series.free_pixel_data()

    @property
    def slice_thickness(self):
        return self.spacing[0]

    @property
    def spacing(self):
        return [
            *self.axial_series.true_spacing,
            self.axial_series.slice_thickness
        ]

    @property
    def starting_z_pos(self):
        return self.axial_series.starting_z_pos

    @property
    def z_range_pair(self):
        return self.axial_series.z_range_pair

    @property
    def number_of_dicoms(self):
        return self.pixel_data.shape[0]

    @property
    def series_path(self):
        return "Reconstruction:" + str(self.axial_series.series_path)

    @property
    def resolution(self):
        return self.pixel_data.shape[1], self.pixel_data.shape[2]


@attr.s(frozen=True)
class L3AxialSliceMetadata:
    sagittal_start_z_pos = attr.ib()
    first_axial_pos = attr.ib()
    last_axial_pos = attr.ib()
    l3_axial_image_index = attr.ib()
    axial_image_count = attr.ib()
    predicted_z_position = attr.ib()
    l3_axial_image_dcm_path = attr.ib()

    def as_csv_row(self):
        return [
            self.sagittal_start_z_pos,
            self.predicted_z_position,
            self.first_axial_pos,
            self.last_axial_pos,
            self.l3_axial_image_index,
            self.axial_image_count,
            self.l3_axial_image_dcm_path,
        ]


@attr.s(frozen=True)
class Subject:
    path = attr.ib()

    @property
    def id_(self):
        return self.path.name.split('_')[-1]

    def find_series(self):
        for accession_path in self.path.iterdir():
            for series_path in accession_path.iterdir():
                yield ImageSeries(
                    subject=self,
                    series_path=series_path,
                    accession_path=accession_path
                )


@attr.s(frozen=True)
class NoSubjectDirSubject:
    path = attr.ib()

    @property
    def id_(self):
        return self.path.name.split('-')[0]

    def find_series(self):
          for series_path in self.path.iterdir():
              yield ImageSeries(
                  subject=self,
                  series_path=series_path,
                  accession_path=None
              )

def find_subjects(dataset_dir, new_tim_dir_structure=False):
    subject_class = NoSubjectDirSubject if new_tim_dir_structure else Subject

    for subject_path in Path(dataset_dir).iterdir():
        yield subject_class(path=subject_path)


def find_series(subject):
    warnings.warn("deprecated, use subject.find_series() instead", DeprecatedWarning)
    subject.find_series()


def create_sagittal_mips_from_study_images(study_images):
    # done here so you can require this module w/o loading tensorflow
    from l3finder.preprocess import create_sagittal_mip
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

    sag_filter = functools.partial(
        same_orientation,
        orientation='sagittal',
        excluded_series=excluded_series
    )
    axial_filter = functools.partial(
        same_orientation,
        orientation='axial',
        excluded_series=excluded_series
    )

    def pool_filter(pool, func, candidates):
        return [
            c for c, keep
            in zip(candidates, pool.map(func, candidates))
            if keep
        ]

    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        sagittal_series = pool_filter(p, sag_filter, series)
        axial_series = pool_filter(p, axial_filter, series)

    axial_series = [a_s for a_s in axial_series if a_s.number_of_dicoms > 20]

    return sagittal_series, axial_series, excluded_series


def same_orientation(series, orientation, excluded_series):
    try:
        return series.orientation == orientation
    except UnknownOrientation as e:
        excluded_series.append(e)
        return False


def load_series_to_skip_pickle_file(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def remove_series_to_skip(series_to_skip, input_series):
    series_paths_to_skip = set(s.series_path for s, _ in series_to_skip)

    return [s for s in input_series if s.series_path not in series_paths_to_skip]


def construct_series_for_subjects_without_sagittals(
    subjects,
    sagittal_series,
    axial_series
):
    set_of_subjects_with_sagittals = set(s.subject for s in sagittal_series)

    subjects_without_sagittal = set(
        s
        for s
        in subjects
        if s not in set_of_subjects_with_sagittals
    )

    print("FILTERING OUT 0.5 axials for recons for debugging!")

    def axial_series_is_adequate(series, thickness_mm=0.5):
        try:
            return (series.slice_thickness != thickness_mm and
                    series.number_of_dicoms > 20)
        except AttributeError:
            return False

    axials_to_construct_with = (
        series
        for series
        in axial_series
        if series.subject in subjects_without_sagittal and axial_series_is_adequate(series)
    )

    return [
        ConstructedImageSeries(axial_series=s)
        for s
        in axials_to_construct_with
    ]


def _construct_sagittal_from_axial_image(axial_image):
    return np.flip(np.rot90(np.rot90(axial_image, axes=(0,2)), axes=(1,2), k=3), axis=2)


def filter_axial_series(axial_series):
    def meets_criteria(ax):
        try:
            return all([
                ax.slice_thickness in [3.0, 5.0],
                'lung' not in ax.series_path.name.lower(),
            ])
        except AttributeError:
            return False
    # Must be 5.0 or 3.0 slice thickness for now
    return [ax for ax in axial_series if meets_criteria(ax)]
