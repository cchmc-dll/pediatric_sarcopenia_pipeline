import itertools
from argparse import ArgumentParser

import toolz
from matplotlib import pyplot as plt

from L3_finder.ingest import find_subjects, find_series
from L3_finder.predict import make_predictions_for_images
from L3_finder.preprocess import create_sagittal_mip, preprocess_images


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        'dicom_dir',
        help='Root directory containing dicoms in format output by Tim\'s '
             'script '
    )
    parser.add_argument(
        'dataset_manifest_path',
        help='CSV outlining which series and slices for a subject id'
    )
    parser.add_argument(
        'model_path',
        help='Path to .h5 model'
    )
    parser.add_argument(
        'dataset_cache_path',
        help='.npz path where intermediate dataset can be stored'
    )

    return parser.parse_args()


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


def main():
    args = parse_args()

    l3_images = find_l3_images(args)

    plot_l3_images(l3_images)


def find_l3_images(args):
    subjects = list(find_subjects(args.dicom_dir))
    series = list(flatten(find_series(s) for s in subjects))
    sagittal_series = list(
        filter(lambda s: s.orientation == 'sagittal', series))
    axial_series = list(filter(lambda s: s.orientation == 'axial', series))
    mips = (create_sagittal_mip(series.pixel_data) for series in
            sagittal_series)
    spacings = (series.spacing for series in sagittal_series)
    preprocessed_images = preprocess_images(images=mips, spacings=spacings)
    prediction_results = make_predictions_for_images(
        preprocessed_images,
        model_path=args.model_path
    )
    l3_images = build_l3_images(axial_series, sagittal_series,
                                prediction_results)
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
    l3_images = [
        L3Image(
            sagittal_series=sag,
            axial_series=ax,
            prediction_result=result)
        for ax, (sag, result) in axials_with_sagittals_and_results
    ]
    return l3_images


def plot_l3_images(l3_images):
    for l3_image in l3_images:
        try:
            f, axarr = plt.subplots(1, 2)
            axarr[0].imshow(l3_image.pixel_data)
            axarr[1].imshow(l3_image.prediction_result.display_image)
            plt.show()
        except IndexError:
            print(
                "Prediction: {predicted_y}cm is out of bounds of image for "
                "subject_id: {id_}".format(
                    predicted_y=l3_image.prediction_result.prediction.predicted_y_in_px,
                    id_=l3_image.axial_series.subject.id_
                )
            )
            plt.imshow(l3_image.prediction_result.display_image)
            plt.show()



if __name__ == "__main__":
    main()
