from argparse import ArgumentParser
from collections import namedtuple
import glob
import itertools
from pathlib import Path
import sys

import imageio
import numpy as np
import toolz

from l3finder.ingest import find_subjects, find_series


def main(argv):
    """sma is skeletal muscle area, sum of mask pixels * pixel area"""
    args = parse_args(argv)
    smas = calculate_sma(args)

    print("subject_id,area_mm2")
    for subject_id, area_mm2 in smas:
        print("{},{}".format(subject_id, area_mm2))


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
        '--segmentation_dir',
        required=True,
        help="Directory containing the segmentations in format"
             "[STUDY_ID]/[--prediction_image_name]"
    )

    parser.add_argument(
        '--segmentation_image_name',
        required=True,
        help='Name of the segmentation image, include extension'
             'ex: predition.tif'
    )

    return parser.parse_args(argv)


def calculate_sma(args):
    axial_series = _find_axial_series(args.dicom_dir)
    segmentations = _find_segmentation_images(
        args.segmentation_dir, args.segmentation_image_name
    )
    return _calculate_smas(axial_series, segmentations)


def _find_axial_series(dicom_dir):
    subjects = find_subjects(dicom_dir)
    all_series = _flatten(map(find_series, subjects))
    for s in all_series:
        try:
            if s.orientation == 'axial':
                yield s
        except AttributeError as e:
            print("AttributeError for subject_id:", s.subject.id_, file=sys.stderr)
    # return filter(lambda s: s.orientation == 'axial', all_series)


def _flatten(sequence):
    return itertools.chain(*sequence)


SegmentationImage = namedtuple("SegmentationImage", ["subject_id", "image_data"])


def _find_segmentation_images(segmentation_dir, image_name):
    for image_path in glob.glob("{}/**/{}".format(segmentation_dir, image_name)):
        yield SegmentationImage(
            subject_id=_subject_id_from_seg_image_path(image_path),
            image_data=imageio.imread(image_path),
        )


def _subject_id_from_seg_image_path(image_path):
    return Path(image_path).parent.name


SegmentationArea = namedtuple("SegmentationArea", ["subject_id", "area_mm2"])


def _calculate_smas(axial_series, segmentations):
    segmentation_series_pairs = toolz.join(
        leftkey=lambda s: s.subject_id,
        leftseq=segmentations,
        rightkey=lambda ax: ax.subject.id_,
        rightseq=axial_series,
    )

    for segmentation, series in segmentation_series_pairs:
        try:
            scale_factor = np.product(np.array(series.resolution) / np.array(segmentation.image_data.shape))
            segmented_pixels = np.count_nonzero(segmentation.image_data)
            pixel_area = np.product(series.true_spacing)
            yield SegmentationArea(
                subject_id=series.subject.id_,
                area_mm2=pixel_area * segmented_pixels * scale_factor
            )
        except AttributeError as e:
            print("AttributeError for subject_id:", series.subject.id_, "when calculating area", file=sys.stderr)

if __name__ == "__main__":
    main(sys.argv[1:])
