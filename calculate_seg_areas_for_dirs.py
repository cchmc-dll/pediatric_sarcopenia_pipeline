from argparse import ArgumentParser
from collections import namedtuple
import csv
from pathlib import Path
import sys

import calculate_segmentation_area_for_dir

def main(argv):
    """sma is skeletal muscle area, sum of mask pixels * pixel area"""
    args = parse_args(argv)
    areas_by_directory = calculate_areas(args)
    output_to_csv(args.output_directory_path, areas_by_directory)


def parse_args(argv):
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
        '--segmentation_dirs',
        required=True,
        nargs="+",
        help="Directories containing the segmentations in format"
             "[STUDY_ID]/[--prediction_image_name]"
    )

    parser.add_argument(
        '--segmentation_image_name',
        required=True,
        help='Name of the segmentation image, include extension'
             'ex: predition.tif'
    )

    parser.add_argument(
        '--output_directory_path',
        required=True,
        help="Path to directory where CSV files are going to go. Must exist."
    )


    return parser.parse_args(argv)


AreaArgs = namedtuple("AreaArgs", "dicom_dir, segmentation_dir, segmentation_image_name")
Result = namedtuple("Result", "segmentation_dir areas")


def calculate_areas(args):
    for segmentation_dir in args.segmentation_dirs:
        area_args = AreaArgs(
            dicom_dir=args.dicom_dir,
            segmentation_dir=segmentation_dir,
            segmentation_image_name=args.segmentation_image_name
        )
        yield Result(
            segmentation_dir=segmentation_dir,
            areas=calculate_segmentation_area_for_dir.calculate_sma(area_args),
        )


def output_to_csv(output_directory_path, areas_by_directory):
    for segmentation_dir, areas in areas_by_directory:
        filename = Path(segmentation_dir).name + ".csv"
        output_csv_path = Path(output_directory_path, filename)
        with open(output_csv_path, "w") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["subject_id", "area_mm2"])
            for area in areas:
                csv_writer.writerow(area)
        print(output_csv_path)





if __name__ == "__main__":
    main(sys.argv[1:])
