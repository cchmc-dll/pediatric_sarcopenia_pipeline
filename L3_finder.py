from argparse import ArgumentParser
from pathlib import Path
import numpy as np

from L3_finder.images import find_images_and_metadata
from L3_finder.preprocess import process_image


def main():
    args = parse_args()
    ensure_output_path_exists(args)

    data_for_l3_finder = find_images_and_metadata(
        manifest_csv=Path(args.dicom_csv),
        dataset_path=Path(args.dicom_dir),
        intermediate_nifti_dir=Path(args.nifti_dir))

    # create_mip_from_path(Path("D:/muscle_segmentation/dataset_2_nifti/nifti_out/"))
    # sagittal_paths = list(Path(args.input_dir).glob('*.nii'))
    # images = np.array([process_image(i) for i in sagittal_paths])
    #
    np.savez_compressed(args.output_path, **data_for_l3_finder)
    print(args.output_path)


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('dicom_dir', help='Root directory containing dicoms in format output by Tim\'s script')
    parser.add_argument('dicom_csv', help='CSV outlining which series and slices for a subject id')
    parser.add_argument('nifti_dir', help='Dir for intermediately created niftis')
    parser.add_argument('output_path', help='output .npz file path')

    return parser.parse_args()


def ensure_output_path_exists(args):
    if not Path(args.output_path).parent.exists():
        raise FileNotFoundError(args.output_path)


if __name__ == '__main__':
    main()

