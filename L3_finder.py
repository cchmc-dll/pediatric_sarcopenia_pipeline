import random
from argparse import ArgumentParser
from L3_finder.images import find_images_and_ydata_in_l3_finder_format, CachableL3ImageLoaderStep
from L3_finder.preprocess import slice_middle_images, create_mip
from L3_finder.predict import make_predictions_for_images
from pathlib import Path
import numpy as np
from toolz import pipe
from matplotlib import pyplot as plt
import imageio

from ct_slice_detection.io.preprocessing import to256
from util.pipelines import make_load_from_cache_or_run_step


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('dicom_dir', help='Root directory containing dicoms in format output by Tim\'s script')
    parser.add_argument('dataset_manifest_path', help='CSV outlining which series and slices for a subject id')
    parser.add_argument('model_path', help='Path to .h5 model')
    parser.add_argument('dataset_cache_path', help='.npz path where intermediate dataset can be stored')

    return parser.parse_args()


def main():
    args = parse_args()
    # dataset = find_images_and_ydata_in_l3_finder_format(args.dataset_manifest_path, Path(args.dicom_dir))

    dataset = make_load_from_cache_or_run_step(
        use_cache=True,
        pipeline_step=CachableL3ImageLoaderStep(
            cached_file_path=Path(args.dataset_cache_path),
            manifest_csv_path=Path(args.dataset_manifest_path),
            dataset_path=Path(args.dicom_dir)
        )
    )()

    predictions = make_predictions_for_images(dataset, args.model_path)

    random.seed(9000)

    for index, result in enumerate(random.sample(list(predictions), 10)):
        imageio.imwrite(f'{index}.png', result.display_image)

        print(result.prediction.predicted_y)
        pass

        # i = result.display_image
        # debug_plot(i, shape=(i.shape[1], i.shape[2]))


def debug_plot(image, shape):
    plt.imshow(image.reshape(shape))
    plt.show()

# def debug_plot(images):
#     from matplotlib import pyplot as plt
#     # fig=plt.figure(figsize=(10, 10))
#     # columns = 4
#     # rows = 3
#     # for i in range(1, columns*rows +1):
#     #     fig.add_subplot(rows, columns, i)
#     plt.imshow(images[0])
#     plt.show()

if __name__ == "__main__":
    main()