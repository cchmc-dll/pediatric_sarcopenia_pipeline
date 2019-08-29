from argparse import ArgumentParser
from pathlib import Path
import numpy as np

from L3_finder.images import find_images_and_ydata_in_l3_finder_format, CachableL3ImageLoaderStep
from util.pipelines import make_load_from_cache_or_run_step


def main():
    args = parse_args()
    ensure_output_path_exists(args)

    data_for_l3_finder = make_load_from_cache_or_run_step(
        use_cache=True,
        pipeline_step=CachableL3ImageLoaderStep(
            cached_file_path=Path(args.output_path),
            manifest_csv_path=Path(args.dicom_csv),
            dataset_path=Path(args.dicom_dir)
        )
    )()

    print(args.output_path)


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('dicom_dir', help='Root directory containing dicoms in format output by Tim\'s script')
    parser.add_argument('dicom_csv', help='CSV outlining which series and slices for a subject id')
    parser.add_argument('output_path', help='output .npz file path')

    return parser.parse_args()


def ensure_output_path_exists(args):
    if not Path(args.output_path).parent.exists():
        raise FileNotFoundError(args.output_path)


if __name__ == '__main__':
    main()

