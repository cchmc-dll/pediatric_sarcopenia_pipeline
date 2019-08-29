import functools
import random
from argparse import ArgumentParser
from pathlib import Path

import toolz
from matplotlib import pyplot as plt

from L3_finder.images import LoadL3DatasetCachableStep
from L3_finder.predict import make_predictions_for_images
from util.pipelines import build_callable_that_loads_from_cache_or_runs_step


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


def main():
    args = parse_args()

    study_images = load_l3_dataset(args)

    pipeline = build_l3_finder_pipeline(args)
    predictions = pipeline(study_images)


def load_l3_dataset(args):
    return build_callable_that_loads_from_cache_or_runs_step(
        pipeline_step=LoadL3DatasetCachableStep(
            cached_file_path=Path(args.dataset_cache_path),
            manifest_csv_path=Path(args.dataset_manifest_path),
            dataset_path=Path(args.dicom_dir)
        ),
        use_cache=True
    )()


def build_l3_finder_pipeline(args):
    def l3_finder_pipeline(data):
        return toolz.pipe(
            data,
            functools.partial(
                make_predictions_for_images,
                model_path=args.model_path
            ),
            list,
            show_sample_of_prediction_images
        )

    return l3_finder_pipeline


def show_sample_of_prediction_images(predictions):
    random.seed(1000)

    enumerated_predictions_sample = enumerate(
        random.sample(
            predictions,
            min(10, len(predictions))
        )
    )

    for index, result in enumerated_predictions_sample:
        plt.imshow(result.display_image, cmap='bone')
        plt.show()

    return predictions


if __name__ == "__main__":
    main()
