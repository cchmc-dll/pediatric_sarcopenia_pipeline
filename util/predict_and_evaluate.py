from argparse import ArgumentParser
import sys
import glob
from pathlib import Path
from os.path import basename, splitext, join
import subprocess

def main(argv):
    args = parse_args(argv)

    models = list(glob.glob(args.model_glob_pattern, recursive=True))

    print("Using these models for prediction:")
    for model in models:
        print(model, file=sys.stderr)

    prediction_dirs = run_prediction(args, models)
    run_evaluation(args, prediction_dirs)


def parse_args(argv):
    parser = ArgumentParser()

    parser.add_argument("model_glob_pattern", help="pattern to glob for models. MUST BE ABSOLUTE PATH")
    parser.add_argument("data_file", help="h5 file with the images")
    parser.add_argument("testing_split", help="pkl file with indicies to test")
    parser.add_argument("dir_for_results", help="Dir where the results are going to go")

    return parser.parse_args(argv)


def run_prediction(args, models):
    prediction_dirs = []
    for model_path in models:
        folder_name = splitext(basename(model_path))[0]
        prediction_folder = join(args.dir_for_results, folder_name)

        prediction_dirs.append(prediction_folder)

        cmd = [
            "python", "run_prediction.py",
            "--training_model_name", model_path,
            "--data_file", args.data_file,
            "--testing_split", args.testing_split,
            "--problem_type", "Segmentation",
            "--prediction_folder", prediction_folder,
            "--image_masks", "Muscle",
        ]
        subprocess.check_call(cmd)

    return prediction_dirs



def run_evaluation(args, prediction_dirs):
    for pred_dir in prediction_dirs:
        cmd = [
            "python", "run_evaluation_onTIF.py",
            "--input_dir", pred_dir,
            "--output_file", f"{basename(pred_dir)}.csv"
        ]
        subprocess.check_call(cmd)


if __name__ == "__main__":
    main(sys.argv[1:])

# python run_prediction.py --training_model_name=/Users/jamescastiglione/research/ms/combined_2020_02_19_actually_bin_cross_with_correct_splits_20200219-132849455285/combined_actually_bin_cross_219_fold_0 --data_file=/Users/jamescastiglione/research/combined_205_fixed_checked_2020-02-18.h5 --testing_split=/Users/jamescastiglione/research/corrected_kfold/combined_205_fixed_checked_2020-02-18_kfold/fold_0_test.pkl --problem_type=Segmentation --prediction_folder=/Users/jamescastiglione/research/segmentation_predictions/bin_cross_fold_0 --image_masks=Muscle
