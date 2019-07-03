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

import tensorflow as tf
sess = tf.Session()


config = dict()

# General Parameters

config["data_file"] = "CT2_with_norm_changes.h5"
config["model_images"] = "Unet2DBN_muscle_wdscloss.h5"
config['prediction_folder'] = 'CT2_norm'
config["testing_split"] = 'validation_norm_1.0' +'.pkl'



config["training_model"] = config["model_images"]

config["monitor"] = 'output'

# Clinical parameters
config["overwrite"] = 0


# Image specific parameters
config["image_shape"] = (256, 256)  # This determines what shape the images will be cropped/resampled to.
config["n_channels"] = 1            # All image channels that will be used as input, image_mask can be input for classification problems and output for segmentation problems.


config["input_type"] = "Image"
config["test_model"] = config["model_images"]

config["batch_size"] = 5
config["validation_batch_size"] = config['batch_size']
config["GPU"] = 1
config["CPU"] = 4
config['patch_shape'] = None
config['skip_blank'] = False


config["labels"] = (1,)  # the label numbers on the input image
config["all_modalities"] = ["CT"]
config["training_modalities"] = config["all_modalities"]
config["threshold"] = 0.5
config["problem_type"] = "Segmentation"


def main(overwrite=False):
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
        data_file = tables.open_file(os.path.abspath(os.path.join('datasets',config["data_file"])),mode='r')
    except:
        raise Exception("Error: Could not open data file, check if config[\"data_file\"] is defined \n")

    # Step 3: LOAD DATA
    testing_file = os.path.abspath(os.path.join('datasets',config['testing_split']))
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
    output_dir = os.path.abspath(os.path.join('predictions',config["prediction_folder"])) 
    for index in test_list:
        if 'subject_ids' in data_file.root:
            case_directory = os.path.join(output_dir, data_file.root.subject_ids[index].decode('utf-8'))
        else:
            case_directory = os.path.join(output_dir, "validation_case_{}".format(index))

        run_validation_case(data_index=index, output_dir=case_directory, model=model, data_file=data_file,
                            training_modalities=config["training_modalities"], output_label_map=True, labels=config["labels"],
                            threshold=config["threshold"])

    

if __name__ == "__main__":
    main(overwrite=config["overwrite"])