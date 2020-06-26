from matplotlib import pyplot as plt

from argparse import ArgumentParser
import itertools
import sys

import attr
import toolz


from l3finder.ingest import find_subjects, separate_series, \
    load_series_to_skip_pickle_file, remove_series_to_skip, \
    construct_series_for_subjects_without_sagittals
from l3finder.output import output_l3_images_to_h5, output_images
from util.reify import reify


@attr.s
class SagittalMIP:
    series = attr.ib()
    preprocessed_image = attr.ib()

    @property
    def subject_id(self):
        return self.series.subject.id_


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        '--dicom_dir',
        required=True,
        help='Root directory containing dicoms in format output by Tim\'s '
             'script. That is subject_1/accession_xyz/series{sagittal & '
             'axial}. The accession directory should contain both a sagittal '
             'preprocessed_image series and an axial preprocessed_image series. '
    )
    parser.add_argument(
        '--new_tim_dicom_dir_structure',
        action='store_true',
        help='Gets subjects in the format used for the 10000 pt dataset versus that used for the 380 pt dataset'
    )
    parser.add_argument(
        '--model_path',
        required=True,
        help='Path to .h5 model trained using '
             'https://github.com/fk128/ct-slice-detection Unet model. '
    )

    parser.add_argument(
        '--output_directory',
        required=True,
        help='Path to directory where output files will be saved. Will be '
             'created if it does not exist '
    )

    parser.add_argument(
        '--show_plots',
        action='store_true',
        help='Path to directory where output files will be saved'
    )

    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite files within target folder'
    )

    parser.add_argument(
        '--save_plots',
        action='store_true',
        help='If true, will save side-by-side plot of predicted L3 and the '
             'axial slice at that level '
    )

    parser.add_argument(
        "--series_to_skip_pickle_file",
        help="Pickle file of series dumped from identify_broken_dicoms.py"
    )

    parser.add_argument(
        '--cache_intermediate_results',
        action='store_true',
        help='If true, will cache the results of some of the longer running '
             'computations in the passed --cache_dir. Note: there is no check to make'
             'sure that you actually passed a dir'
    )

    parser.add_argument(
        "--cache_dir",
        help="Directory to store cached files in. If none given, no caching"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    l3_images = find_l3_images(args)

    print("Outputting images")
    output_images(
        l3_images,
        args=dict(
            output_directory=args.output_directory,
            should_plot=args.show_plots,
            should_overwrite=args.overwrite,
            should_save_plots=args.save_plots
        )
    )


def find_l3_images(args):
    print("Finding subjects")

    subjects = list(
        find_subjects(
            args.dicom_dir,
            new_tim_dir_structure=args.new_tim_dicom_dir_structure
        )
    )

    print("Finding series")

    series = list(flatten(s.find_series() for s in subjects))

    print("Separating series")
    sagittal_series, axial_series, excluded_series = separate_series(series)
    print("SHORTENING for development")
    sagittal_series = sagittal_series[:35]
    axial_series = axial_series[:35]
    # investigate = set(["56"])
    # sagittal_series = [s for s in sagittal_series if s.subject.id_ in investigate]
    # axial_series = [s for s in axial_series if s.subject.id_ in investigate]
    constructed_sagittals = construct_series_for_subjects_without_sagittals(
        subjects, sagittal_series, axial_series
    )

    import ipdb; ipdb.set_trace()

    print(
        "Series separated\n",
        len(sagittal_series), "sagittal series.",
        len(axial_series), "axial series.",
        len(excluded_series), "excluded series."
    )

    if args.series_to_skip_pickle_file:
        print("Removing unwanted series")
        series_to_skip = load_series_to_skip_pickle_file(
            args.series_to_skip_pickle_file)
        sagittal_series = remove_series_to_skip(series_to_skip, sagittal_series)
        axial_series = remove_series_to_skip(series_to_skip, axial_series)

    print("Importing things that need tensorflow...")
    from l3finder.predict import make_predictions_for_sagittal_mips
    from l3finder.preprocess import create_sagittal_mip, preprocess_images, \
        group_mips_by_dimension, create_sagittal_mips_from_series

    print("Creating sagittal MIPS")
    mips = create_sagittal_mips_from_series(
        many_series=sagittal_series,
        cache_dir=args.cache_dir,
        cache=args.cache_intermediate_results,
    )
    spacings = [series.spacing for series in sagittal_series]

    print("Preprocessing Images")
    preprocessed_images = preprocess_images(images=mips, spacings=spacings)

    sagittal_mips = [
        SagittalMIP(s, i)
        for s, i
        in zip(sagittal_series, preprocessed_images)
    ]

    print("Separate heights for better batching")
    mips_by_dimension = group_mips_by_dimension(sagittal_mips)
    print("Dimensions in set:", mips_by_dimension.keys())

    print("Making predictions")
    prediction_results = []
    for dimension, sagittal_mips in mips_by_dimension.items():
        dim_group_results = make_predictions_for_sagittal_mips(
            sagittal_mips,
            model_path=args.model_path,
            shape=dimension,
        )
        prediction_results.extend(dim_group_results)
    l3_images = build_l3_images(axial_series, prediction_results)
    return l3_images


def flatten(sequence):
    return itertools.chain(*sequence)


def build_l3_images(axial_series, prediction_results):
    axials_with_prediction_results = toolz.join(
        leftkey=lambda ax: ax.subject.id_,
        leftseq=axial_series,
        rightkey=lambda pred_res: pred_res.input_mip.subject_id,
        rightseq=prediction_results
    )
    l3_images = [
        L3Image(
            sagittal_series=result.input_mip.series,
            axial_series=ax,
            prediction_result=result)
        for ax, result in axials_with_prediction_results
    ]
    return l3_images


@attr.s
class L3Image:
    axial_series = attr.ib()
    sagittal_series = attr.ib()
    prediction_result = attr.ib()

    @property
    def pixel_data(self):
        return self.axial_series.image_at_pos_in_px(
            self.prediction_result.prediction.predicted_y_in_px,
            sagittal_start_z_pos=self.sagittal_series.starting_z_pos
        )

    @property
    def height_of_sagittal_image(self):
        return self.sagittal_series.resolution[0]

    @property
    def number_of_axial_dicoms(self):
        return self.axial_series.number_of_dicoms

    @property
    def subject_id(self):
        return self.axial_series.subject.id_

    @reify
    def prediction_index(self):
        index, metadata =  self.axial_series.image_index_at_pos(
            self.prediction_result.prediction.predicted_y_in_px,
            sagittal_start_z_pos=self.sagittal_series.starting_z_pos
        )
        return index, metadata

    def as_csv_row(self):
        prediction = self.prediction_result.prediction
        prediction_index, l3_axial_slice_metadata = self.prediction_index
        row = [
            self.axial_series.id_,
            prediction.predicted_y_in_px,
            prediction.probability,
            self.sagittal_series.series_path,
            self.axial_series.series_path,
        ]
        row.extend(l3_axial_slice_metadata.as_csv_row())
        return row

    @property
    def predicted_y_in_px(self):
        return self.prediction_result.prediction.predicted_y_in_px

    def free_pixel_data(self):
        """Frees the memory used in the underlying ImageSeries objects"""
        self.axial_series.free_pixel_data()
        self.sagittal_series.free_pixel_data()


if __name__ == "__main__":
    main()
