import csv
import sys
from collections import defaultdict
from pathlib import Path

import imageio
import numpy as np
import attr
import SimpleITK as sitk

import run_sma_experiment
from L3_finder import L3Image
from l3finder import ingest


def main(argv):
    args = parse_args(argv)
    segmentations, average_masks, smas = make_comparison(args)
    output_comparison(args["output_dir"], segmentations, average_masks, smas)
    return segmentations, average_masks, smas


def parse_args(argv):
    return {
        "l3_prediction_dir": "/Users/jamescastiglione/research/l3_finder_results/results_from_correct_indices/child_9_slice_w_adult_weights",
        "model_name": "UNet1D",
        "num_folds": 5,
        "dicom_dataset_dir": "/Users/jamescastiglione/git/jac241/Muscle_Segmentation/datasets/combined_dataset",
        "seg_models_dir": "/Users/jamescastiglione/research/final_seg_models_dice",
        "output_dir": "/Users/jamescastiglione/research/smi_results/pipeline_predictions"
    }


def make_comparison(args):
    all_predictions = load_l3_predictions(args['l3_prediction_dir'], args['model_name'], args['num_folds'])
    mean_predictions = calc_mean_predictions(all_predictions)
    subjects_w_preds = find_subjects_w_preds(mean_predictions, list(ingest.find_subjects(args['dicom_dataset_dir'])))
    l3_images = load_l3_images_from_predictions(mean_predictions, subjects_w_preds)

    configs = seg_model_configs(args['seg_models_dir'])
    segmentations = do_segmentation_for_each_config(configs, l3_images)
    average_masks = calculate_average_masks(segmentations)
    smas = calc_smas(average_masks, segmentations)
    return segmentations, average_masks, smas


subject_id_col = 0
pred_in_px_col = 2


def load_l3_predictions(l3_prediction_dir, model_name, num_folds):
    csv_filename_template = f"{model_name}_cv_{{}}_of_{num_folds}_preds.csv"
    predictions = defaultdict(list)

    for fold_index in range(1, 6):
        with open(Path(l3_prediction_dir, csv_filename_template.format(fold_index))) as csvfile:
            reader = csv.reader(csvfile)
            next(reader)

            for row in reader:
                predictions[row[subject_id_col]].append(float(row[pred_in_px_col]))

    return predictions


def calc_mean_predictions(all_predictions: defaultdict):
    result = {}
    for subject_id, prediction_list in all_predictions.items():
        result[subject_id] = np.mean(prediction_list)
    return result


def find_subjects_w_preds(predictions, all_subjects):
    subject_ids_w_preds = set(predictions.keys())
    return [s for s in all_subjects if s.id_ in subject_ids_w_preds]


@attr.s
class MinimalPrediction:
    predicted_y_in_px = attr.ib()


@attr.s
class MinimalResult:
    prediction = attr.ib()


def load_l3_images_from_predictions(mean_predictions, subjects_w_preds):
    l3_images = []

    for subject in subjects_w_preds:
        all_series = list(subject.find_series())
        sagittal_series = next(s for s in all_series if s.orientation == 'sagittal')
        axial_series = next(s for s in all_series if s.orientation == 'axial')
        l3_images.append(
            L3Image(
                axial_series=axial_series,
                sagittal_series=sagittal_series,
                prediction_result=MinimalResult(
                    MinimalPrediction(
                        predicted_y_in_px=mean_predictions[subject.id_]
                    )
                )
            )
        )

    return l3_images


def seg_model_configs(seg_models_dir):
    result = []
    model_paths = sorted(Path(seg_models_dir).glob("*.h5"))
    for model_path in model_paths:
        result.append({"model_path": str(model_path)})

    return result


def do_segmentation_for_each_config(seg_model_config, l3_images):
    segmentations_for_each_model = []
    for config in seg_model_config:
        print("segmenting for model", config)
        segmentations_for_each_model.append(
            run_sma_experiment.segment_muscle(config, l3_images)
        )
    return segmentations_for_each_model


def calculate_average_masks(segmentations):
    mask_lists = [s.reshaped_masks for s in segmentations]
    average_masks = []
    for one_subjects_masks in zip(*mask_lists):
        sitk_images = [sitk.Cast(sitk.GetImageFromArray(mask), sitk.sitkUInt8)
                       for mask in one_subjects_masks]
        result = sitk.LabelVoting(sitk_images, 1)
        arr = sitk.GetArrayFromImage(result)
        average_masks.append(arr)

    return average_masks


def calc_smas(average_masks, segmentations):
    output = []
    for index, mask in enumerate(average_masks):
        output.append(
            run_sma_experiment.calculate_sma_for_series_and_mask(
                series=segmentations[0].l3_images[index].axial_series,
                mask=mask,
            )
        )
    return output


def output_comparison(output_dir, segmentations, average_masks, smas):
    for mask, sma, l3_image in zip(average_masks, smas,
                                   segmentations[0].l3_images):
        print(','.join([sma.subject_id, str(sma.area_mm2)]))
        mask_path = Path(output_dir, f"{sma.subject_id}_mask.tif")
        ct_path = Path(output_dir, f"{sma.subject_id}_ct.tif")
        imageio.imsave(str(mask_path), mask * np.iinfo(np.uint8).max)
        imageio.imsave(str(ct_path), l3_image.pixel_data)


if __name__ == "__main__":
    segmentations, average_masks, smas = main(sys.argv[1:])