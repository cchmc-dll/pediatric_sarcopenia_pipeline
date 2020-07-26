from argparse import ArgumentParser
import csv
import json
import multiprocessing
import os

import attr
import cv2
from imageio import imsave
from keras.models import load_model
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk

from L3_finder import find_l3_images
from unet3d.metrics import (dice_coefficient, dice_coefficient_loss, dice_coef, dice_coef_loss,dice_coefficient_monitor,
                            weighted_dice_coefficient_loss_2D, weighted_dice_coefficient_2D)
from l3finder.output import output_l3_images_to_h5, output_images


def main():
    args = parse_args()
    config = parse_config_file(args)
    print("Config: \n", config)

    l3_images = find_l3_images(config["l3_finder"])

    print("Outputting L3 images")
    output_images(
        l3_images,
        args=dict(
            output_directory=config["l3_finder"]["output_directory"],
            should_plot=config["l3_finder"]["show_plots"],
            should_overwrite=config["l3_finder"]["overwrite"],
            should_save_plots=config["l3_finder"]["save_plots"]
        )
    )
    print("Segmenting muscle...")
    sma_images = segment_muscle(config["muscle_segmentor"], l3_images)
    print("Calculating sma")
    areas = calculate_smas(sma_images)

    print("Outputing SMA results")
    output_sma_results(
        config["muscle_segmentor"]["output_directory"],
        sma_images,
        areas
    )


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("json_config_path", help="path to json config file")

    return parser.parse_args()


def parse_config_file(args):
    with open(args.json_config_path, "r") as f:
        return json.load(f)


@attr.s
class SegmentedImages:
    l3_images = attr.ib()
    l3_ndas = attr.ib()
    tableless_images = attr.ib()
    thresholded_images = attr.ib()
    normalized_images = attr.ib()
    resized_images = attr.ib()
    xs = attr.ib()
    ys = attr.ib()
    masks = attr.ib()
    reshaped_masks = attr.ib()

    def __len__(self):
        return len(self.l3_images)

    def subject_ids(self):
        return [i.subject_id for i in self.l3_images]


def segment_muscle(config, l3_images):
    print("- Loading l3 axial images")
    l3_ndas = load_l3_ndas(l3_images)
    print("- Removing table")
    tableless_images = remove_table(l3_ndas)
    print("- Thresholding images")
    thresholded_images = threshold_images(tableless_images)
    print("- Normalizing images")
    normalized_images = normalize_images(thresholded_images)
    print("- Resizing images")
    resized_images = resize_images(normalized_images)
    xs = reshape_for_model(resized_images)
    model = configure_and_load_model(config["model_path"])
    print("- Predicting segmentation")
    ys = model.predict(xs)
    print("- Converting masks to images segmentation")
    masks = convert_to_images(ys)
    reshaped_masks = reshape_masks(masks)
    return SegmentedImages(l3_images, l3_ndas, tableless_images, thresholded_images,
                           normalize_images, resized_images, xs, ys, masks,
                           reshaped_masks)


def load_l3_ndas(l3_images):
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        ndas = list(tqdm(pool.imap(_load_l3_pixel_data, l3_images)))
        pool.close()
        pool.join()

    return ndas


def _load_l3_pixel_data(l3_image):
    return l3_image.pixel_data


def remove_table(l3_ndas):
    print("  - zeroing images")
    zeroed_images = [
        l3_nda + (l3_nda.min() * -1)
        for l3_nda
        in tqdm(l3_ndas)
    ]

    print("  - removing table")
    with multiprocessing.Pool(multiprocessing.cpu_count() // 2) as pool:
        tableless_images = list(tqdm(pool.imap(_remove_table, zeroed_images)))
        pool.close()
        pool.join()

    # tableless_images = np.empty(shape=(len(l3_ndas), 512, 512))
    # for index, l3_nda in enumerate(l3_ndas):
        # zeroed = l3_nda + (l3_nda.min() * -1)
        # sitk_image = _remove_table(sitk.GetImageFromArray(zeroed))
        # tableless_images[index] = sitk.GetArrayFromImage(sitk_image)
    return np.array(tableless_images)


def _remove_table(CT_nda,l_thresh=1300,h_thresh=3500,seed=[256, 256]):
    CT = sitk.GetImageFromArray(CT_nda)
    #
    # Blur using CurvatureFlowImageFilter
    #
    blurFilter = sitk.CurvatureFlowImageFilter()
    blurFilter.SetNumberOfIterations(5)
    blurFilter.SetTimeStep(0.125)
    image = blurFilter.Execute(CT)

    #
    # Set up ConnectedThresholdImageFilter for segmentation
    #
    segmentationFilter = sitk.ConnectedThresholdImageFilter()
    segmentationFilter.SetLower(float(l_thresh))
    segmentationFilter.SetUpper(float(h_thresh))
    segmentationFilter.SetReplaceValue(1)
    segmentationFilter.AddSeed(seed)

    # Run the segmentation filter
    image = segmentationFilter.Execute(image)
    image[seed] = 1

    # Fill holes
    image = sitk.BinaryFillhole(image);

    # Masking FIlter
    maskingFilter = sitk.MaskImageFilter()
    CT_noTable = maskingFilter.Execute(CT,image)
    return sitk.GetArrayFromImage(CT_noTable)


def threshold_images(images, low=1800, high=2300):
    output = np.copy(images)
    output[output < low] = low
    output[output > high] = high
    return output


def normalize_images(images):
    mean = images.mean()
    std = images.std()
    return (images - mean) / std


def resize_images(images, desired_dims=(256, 256)):
    output = np.empty(shape=(len(images), *desired_dims))
    for index, image in enumerate(images):
        output[index] = cv2.resize(image, desired_dims)
    return output


def reshape_for_model(images):
    return images[:, np.newaxis, :, :]


def configure_and_load_model(model_path):
    custom_objects = {'dice_coefficient_loss': dice_coefficient_loss, 'dice_coefficient': dice_coefficient,
                      'dice_coef': dice_coef, 'dice_coef_loss': dice_coef_loss,'dice_coefficient_monitor': dice_coefficient_monitor,
                      'weighted_dice_coefficient_2D': weighted_dice_coefficient_2D,
                      'weighted_dice_coefficient_loss_2D': weighted_dice_coefficient_loss_2D}

    return load_model(model_path, custom_objects=custom_objects)


def convert_to_images(ys, threshold=0.5):
    output = np.copy(ys)
    output[output > threshold] = 1
    output[output <= threshold] = 0
    return output


def reshape_masks(masks):
    count = masks.shape[0]
    rows = masks.shape[2]
    columns = masks.shape[3]

    return masks.reshape(count, rows, columns)


def calculate_smas(sma_images):
    output = []
    for index in range(len(sma_images)):
        output.append(
            calculate_sma_for_series_and_mask(
                sma_images.l3_images[index].axial_series,
                sma_images.reshaped_masks[index],
            )
        )
    return output


@attr.s
class SegmentationArea:
    subject_id = attr.ib()
    area_mm2 = attr.ib()


def calculate_sma_for_series_and_mask(series, mask):
    scale_factor = np.product(np.array(series.resolution) / np.array(mask.shape))
    segmented_pixels = np.count_nonzero(mask)
    pixel_area = np.product(series.true_spacing)
    return SegmentationArea(
        subject_id=series.subject.id_,
        area_mm2=pixel_area * segmented_pixels * scale_factor
    )


def output_sma_results(output_dir, sma_images, areas):
    os.makedirs(output_dir, exist_ok=True)

    csv_filename = os.path.join(output_dir, "areas-mm2_by_subject_id.csv")
    with open(csv_filename, "w") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["subject_id", "area_mm2", "sagittal_series", "axial_series"])

        for index in range(len(sma_images)):
            l3_image = sma_images.l3_images[index]
            base = os.path.join(output_dir, str(index) + "_" + l3_image.subject_id)
            imsave(base + "_CT.tif", sma_images.tableless_images[index].astype(np.float32))
            imsave(base + "_muscle.tif", sma_images.masks[index][0])

            row = [
                *attr.astuple(areas[index]),
                l3_image.sagittal_series.series_name,
                l3_image.axial_series.series_name,
            ]
            csv_writer.writerow(row)


if __name__ == "__main__":
    main()
