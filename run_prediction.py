import os
import glob
import cmd
from pyimagesearch.nn.conv import *
import numpy as np
import pandas as pd
import tables
from random import shuffle
from pyimagesearch.nn.conv.Unet2D import Unet2D
from unet3d.normalize import normalize_data_storage,normalize_clinical_storage
from unet3d.utils.utils import pickle_dump, pickle_load
from sklearn import preprocessing
from sklearn.metrics import classification_report
import keras
from keras.utils import np_utils
from keras.utils import plot_model
from keras.optimizers import SGD
from pyimagesearch.callbacks import TrainingMonitor
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
from keras.utils.training_utils import multi_gpu_model
from alt_model_checkpoint import AltModelCheckpoint
from imutils import paths
import matplotlib.pyplot as plt
import imutils
import cv2
import os
from pyimagesearch.utils.generator_utils_Elan import *
from keras import Model
from pyimagesearch.callbacks.datagenerator import *
from keras.models import load_model
from unet3d.prediction2D import run_validation_case
from unet3d.metrics import (dice_coefficient, dice_coefficient_loss, dice_coef, dice_coef_loss,dice_coefficient_monitor,
                            weighted_dice_coefficient_loss_2D, weighted_dice_coefficient_2D)

from argparse import ArgumentParser
import tensorflow as tf
sess = tf.Session()


config = dict()

# General Parameters

# config["data_file"] = "/Users/jamescastiglione/research/combined_205_fixed_checked_2020-02-18.h5"
# config["model_images"] = "/Users/jamescastiglione/research/ms/combined_205_fixed_checked_2020-02-18_dice_20200218-130639/combined_2020-02-18_dice_fold_0.h5"
# config['prediction_folder'] = '/Users/jamescastiglione/research/segmentation_predictions/dice_fold_0'
# config["testing_split"] = '/Users/jamescastiglione/research/corrected_kfold/combined_205_fixed_checked_2020-02-18_kfold/fold_0_test.pkl'




# config["monitor"] = 'output'

# # Clinical parameters
# config["overwrite"] = 0


# # Image specific parameters
# config["image_shape"] = (256, 256)  # This determines what shape the images will be cropped/resampled to.
# config["n_channels"] = 1            # All image channels that will be used as input, image_mask can be input for classification problems and output for segmentation problems.


# config["input_type"] = "Image"
# config["test_model"] = config["model_images"]

# config["batch_size"] = 5
# config["validation_batch_size"] = config['batch_size']
# config["GPU"] = 1
# config["CPU"] = 4
# config['patch_shape'] = None
# config['skip_blank'] = False


# config["labels"] = (1,)  # the label numbers on the input image
# config["all_modalities"] = ["CT"]
# config["training_modalities"] = config["all_modalities"]
# config["threshold"] = 0.5
# config["problem_type"] = "Segmentation"

def parse_command_line_arguments():
    parser = ArgumentParser(fromfile_prefix_chars='@')

    req_group = parser.add_argument_group(title='Required flags')
    req_group.add_argument('--training_model_name', required=True, help='Filename of trained model to be saved')
    req_group.add_argument('--data_file', required=True, help='Source of images to predict')
    req_group.add_argument('--testing_split', required=True, help='.pkl file with the indices to test')
    req_group.add_argument('--problem_type', default="Segmentation", required=True, help='Segmentation, Classification, or Regression, default=Segmentation')
    req_group.add_argument('--prediction_folder', required=True, help='Path to directory where preditions files will be saved')
    req_group.add_argument('--image_masks', required=True, help='Comma separated list of mask names, ex: Muscle,Bone,Liver')

    parser.add_argument('--GPU', default=1, type=int, help='Number of GPUs to use, default=1')
    parser.add_argument('--CPU', default=4, type=int, help='Number of CPU cores to use, default=4')
    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument('--patch_shape', default=None)
    parser.add_argument('--skip_blank', default=False, action='store_true')
    parser.add_argument('--input_type', default='Image')
    parser.add_argument('--image_shape', type=tuple, default=(256, 256))
    parser.add_argument('--monitor', default='output', help='directory where monitor output goes')
    parser.add_argument('--overwrite', default=0, type=int, help='0=false, 1=true')
    parser.add_argument('--threshold', default=0.5, type=float, help='Threshold for true on the mask')

    parser.add_argument('--labels', default='1', help='Comma separated list of the label numbers on the input image')
    parser.add_argument('--all_modalities', default='CT', help='Comma separated list of desired image modalities')
    parser.add_argument('--training_modalities', help='Comma separated list of desired image modalities for training only')

    return parser.parse_args()


def build_config_dict(config):
    config["labels"] = tuple(config['labels'].split(','))  # the label numbers on the input image
    config["n_labels"] = len(config["labels"])

    config['all_modalities'] = config['all_modalities'].split(',')

    try:
        config["training_modalities"] = config['training_modalities'].split(',')  # change this if you want to only use some of the modalities
    except AttributeError:
        config["training_modalities"] = config['all_modalities']

    # calculated values from cmdline_args
    config["n_channels"] = len(config["training_modalities"])
    config["input_shape"] = tuple([config["n_channels"]] + list(config["image_shape"]))
    config['image_masks'] = config['image_masks'].split(',')
    config['test_model'] = config['training_model_name']

    return config

def main():
    args = parse_command_line_arguments()
    config = build_config_dict(vars(args))
    run_prediction(config)


def run_prediction(config):
     # Step 1: Check if training type is defined
    try:
        input_type = config["input_type"]
    except:
        raise Exception("Error: Input type not defined | \t Set  config[\"input_type\"] to \"Image\", \"Clinical\" or \"Both\" \n")

    try:
        problem_type = config["problem_type"]
    except:
        raise Exception("Error: Problem type not defined | \t Set  config[\"problem_type\"] to \"Classification\", \"Segmentation\" or \"Regression\" \n")

    # Step 2: Check if the Data File is defined and open it
    try:
        data_file = tables.open_file(os.path.abspath(config["data_file"]), mode='r')
    except:
        raise Exception("Error: Could not open data file, check if config[\"data_file\"] is defined \n")

    # Step 3: LOAD DATA
    testing_file = os.path.abspath(config['testing_split'])
    if data_file.__contains__('/truth'):
        if config["input_type"] is "Both" and data_file.__contains__('/cldata') and data_file.__contains__('/imdata') :
            test_list = pickle_load(testing_file)
        elif config["input_type"] is "Image" and data_file.__contains__('/imdata'):
            test_list = pickle_load(testing_file)
        elif config["input_type"] is "Clinical" and data_file.__contains__('/cldata'):
            test_list = pickle_load(testing_file)
        else:
            print('Input Type: ',input_type)
            print('Clincial data: ', data_file.__contains__('/cldata'))
            print('Image data: ', data_file.__contains__('/imdata'))
            raise Exception("data file does not contain the input group required to train")
    else:
        print('Truth data: ', data_file.__contains__('/truth'))
        raise Exception("data file does not contain the truth group required to train")


    # Step 5: Load Model
    custom_objects = {'dice_coefficient_loss': dice_coefficient_loss, 'dice_coefficient': dice_coefficient,
                      'dice_coef': dice_coef, 'dice_coef_loss': dice_coef_loss,'dice_coefficient_monitor': dice_coefficient_monitor,
                      'weighted_dice_coefficient_2D': weighted_dice_coefficient_2D,
                      'weighted_dice_coefficient_loss_2D': weighted_dice_coefficient_loss_2D}

    model = load_model(config["test_model"], custom_objects=custom_objects)
    print('Test Model')
    print(model.summary())

    # Step 6: Prediction
    output_dir = os.path.abspath(config["prediction_folder"])
    for index in test_list:
        if 'subject_ids' in data_file.root:
            case_directory = os.path.join(output_dir, data_file.root.subject_ids[index].decode('utf-8'))
        else:
            case_directory = os.path.join(output_dir, "validation_case_{}".format(index))

        run_validation_case(data_index=index, output_dir=case_directory, model=model, data_file=data_file,
                            training_modalities=config["training_modalities"], output_label_map=True, labels=config["labels"],
                            threshold=config["threshold"])



if __name__ == "__main__":
    main()
