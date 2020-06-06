import itertools
import sys

from argparse import ArgumentParser

import toolz

from L3_finder.ingest import find_subjects, separate_series
from L3_finder.output import output_l3_images_to_h5, output_images


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        '--dicom_dir',
        required=True,
        help='Root directory containing dicoms in format output by Tim\'s '
             'script. That is subject_1/accession_xyz/series{sagittal & '
             'axial}. The accession directory should contain both a sagittal '
             'image series and an axial image series. '
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

    return parser.parse_args()


def main():
    args = parse_args()

    l3_images = find_l3_images(args)


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

    print(
      "Series separated\n",
      len(sagittal_series), "sagittal series.",
      len(axial_series), "axial series.",
      len(excluded_series), "excluded series."
    )

    print("Importing things that need tensorflow...")
    from L3_finder.predict import make_predictions_for_images
    from L3_finder.preprocess import create_sagittal_mip, preprocess_images

    print("Creating sagittal MIPS")
    mips = (create_sagittal_mip(series.pixel_data) for series in
            sagittal_series)
    spacings = (series.spacing for series in sagittal_series)

    print("Preprocessing Images")
    preprocessed_images = preprocess_images(images=mips, spacings=spacings)

    print("Making predictions")
    prediction_results = make_predictions_for_images(
        preprocessed_images,
        model_path=args.model_path
    )
    l3_images = build_l3_images(
        axial_series, sagittal_series, prediction_results
    )

    return l3_images


def flatten(sequence):
    return itertools.chain(*sequence)


def build_l3_images(axial_series, sagittal_series, prediction_results):
    sagittals_with_results = zip(sagittal_series, prediction_results)
    axials_with_sagittals_and_results = toolz.join(
        leftkey=lambda ax: ax.subject.id_,
        leftseq=axial_series,
        rightkey=lambda sag_with_res: sag_with_res[0].subject.id_,
        rightseq=sagittals_with_results
    )
    l3_images = (
        L3Image(
            sagittal_series=sag,
            axial_series=ax,
            prediction_result=result)
        for ax, (sag, result) in axials_with_sagittals_and_results
    )
    return l3_images


class L3Image(object):
    def __init__(self, axial_series, sagittal_series, prediction_result):
        self.axial_series = axial_series
        self.sagittal_series = sagittal_series
        self.prediction_result = prediction_result

    @property
    def pixel_data(self):
        return self.axial_series.image_at_pos_in_px(
            self.prediction_result.prediction.predicted_y_in_px
        )

    @property
    def subject_id(self):
        return self.axial_series.subject.id_

    @property
    def prediction_index(self):
        return self.axial_series.image_index_at_pos(
            self.prediction_result.prediction.predicted_y_in_px
        )

    def as_csv_row(self):
        prediction = self.prediction_result.prediction
        return [
            self.subject_id,
            prediction.predicted_y_in_px,
            prediction.probability,
            self.sagittal_series.series_path,
            self.axial_series.series_path,
        ]


if __name__ == "__main__":
    main()
