from argparse import ArgumentParser, FileType
from .. import calculate_segmentation_area_for_dir as area
from . import concat_l3_csv_error_mm_column_per_fold as concat
import sys


def main(argv):
    args = parse_args(argv)
    areas_by_directory = area.calculate_areas(args)

    args.csv_files = area.output_to_csv(args, areas_by_directory)

    output_csv_path = concat.concat_csvs(args)
    print(output_csv_path)


def parse_args(argv):
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
        '--segmentation_dirs',
        required=True,
        nargs="+",
        help="Directories containing the segmentations in format"
             "[STUDY_ID]/[--prediction_image_name]"
    )

    parser.add_argument(
        '--segmentation_image_name',
        required=True,
        help='Name of the segmentation preprocessed_image, include extension'
             'ex: predition.tif'
    )

    parser.add_argument(
        '--output_directory_path',
        required=True,
        help="Path to directory where CSV files are going to go. Must exist."
    )

    csv_group = parser.add_argument_group('csv_args')
    csv_group.add_argument('--output_file', required=True, type=FileType('w'))
    csv_group.add_argument('--data_column', required=True)


    return parser.parse_args(argv)

if __name__ == "__main__":
    main(sys.argv[1:])


Input = namedtuple("Input", "seg_dir_wildcard 


"""
python calculate_seg_areas_for_dirs.py
--dicom_dir ~/research/combined_dataset
--segmentation_dirs ~/research/segmentation_predictions/dice_aug/combined_2020-02-18_dice_aug_fold_*
--segmentation_image_name truth.tif
--output_directory_path ~/research/areas/bin_cross/
"""

"""
python util/concat_l3_csv_error_mm_column_per_fold.py
--csv_files /Users/jamescastiglione/research/areas/bin_cross/combined_actually_bin_cross_219_fold_*_prediction.csv 
--output_file ~/research/areas/bin_cross_prediction.csv
--data_column "area_mm2"
"""
