from argparse import ArgumentParser
import itertools
from l3finder.ingest import find_subjects, separate_series
import multiprocessing
import pickle

def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        '--dicom_dir',
        required=True,
        help='Root directory containing dicoms in format output by Tim\'s '
             'script. That is subject_1/accession_xyz/series{sagittal & '
             'axial}. The accession directory should contain both a sagittal '
             'image series and an axial preprocessed_image series. '
    )
    parser.add_argument(
        '--pickle_dump_path',
        required=True,
        help='path to write pickle file with series that throw an error when accessing data'
    )

    return parser.parse_args()

def main():
    args = parse_args()

    l3_images = identify_broken_dicoms(args)


def identify_broken_dicoms(args):
    print("Finding subjects")

    subjects = list(
        find_subjects(
            args.dicom_dir,
            new_tim_dir_structure=True
        )
    )

    print("Finding series")

    series = list(flatten(s.find_series() for s in subjects))

    print("Separating series")
    sagittal_series, axial_series, excluded_series = separate_series(series)

    print("Finding broken dicoms")
    broken_dicoms = []

    with multiprocessing.Pool(48) as pool:
        print("-- Finding sagittals")
        broken_dicoms.extend([s for s in pool.map(series_if_broken, sagittal_series) if s])
        print("-- Finding axials")
        broken_dicoms.extend([s for s in pool.map(series_if_broken, axial_series) if s])

    print("series_id,error")
    for series, error in broken_dicoms:
        print("{},{}".format(series.id_, error))

    with open(args.pickle_dump_path, "wb") as f:
        pickle.dump(broken_dicoms, f)


def flatten(sequence):
    return itertools.chain(*sequence)


def series_if_broken(series):
    try:
        series.pixel_data
        series.free_pixel_data()
        return False
    except Exception as e:
        return (series, e)


if __name__ == "__main__":
    main()
