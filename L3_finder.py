from argparse import ArgumentParser
from L3_finder.images import find_images_and_metadata
from L3_finder.preprocess import slice_middle_images, create_mip
from L3_finder.predict import make_predictions_for_images
from pathlib import Path
import numpy as np
from toolz import pipe

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('dicom_dir', help='Root directory containing dicoms in format output by Tim\'s script')
    parser.add_argument('dataset_manifest_path', help='CSV outlining which series and slices for a subject id')
    parser.add_argument('model_path', help='Path to .h5 model')
    # parser.add_argument('nifti_dir', help='Dir for intermediately created niftis')
    # parser.add_argument('output_path', help='output .npz file path')

    return parser.parse_args()

def main():
    args = parse_args()
    # study_images = find_study_images(Path(args.dicom_dir), args.dataset_manifest_path)
    dataset = find_images_and_metadata(args.dataset_manifest_path, Path(args.dicom_dir))
    # sagittal_mips = [
    #     create_mip(slice_middle_images(image.pixel_data(orientation='sagittal')))
    #     for image
    #     in study_images
    # ]

    debug_plot(dataset['images_s'])

    predictions = make_predictions_for_images(dataset, args.model_path)

def debug_plot(images):
    from matplotlib import pyplot as plt
    # fig=plt.figure(figsize=(10, 10))
    # columns = 4
    # rows = 3
    # for i in range(1, columns*rows +1):
    #     fig.add_subplot(rows, columns, i)
    plt.imshow(images[0])
    plt.show()

if __name__ == "__main__":
    main()