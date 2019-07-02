
# Define Problem Configuration:
import numpy as np
import os
import tables
from random import shuffle

from sklearn.model_selection import train_test_split

from unet3d.utils import pickle_dump, pickle_load

config = dict()
##
config["input_type"] = "Image"
config["image_shape"] = (256,256)
config["input_images"] = "ImageData"

config["overwrite"] = 1
config["problem_type"] = "Segmentation"
config["image_masks"] = ["Muscle"] #["Mask"] #["Label"]   # For Image Masks, will serve as label for segmentation problems
#config["n_channels"] = 1            # All image channels that will be used as input, image_mask can be input for classification problems and output for segmentation problems.

config["normalize"] = False

config["labels"] = (1,)  # the label numbers on the input image
config["n_labels"] = len(config["labels"])

config["all_modalities"] =  ["CT"] #["Input"]
config["training_modalities"] = config["all_modalities"]  # change this if you want to only use some of the modalities
config["n_channels"] = len(config["training_modalities"])
config["input_shape"] = tuple([config["n_channels"]] + list(config["image_shape"]))

##
config["data_file"] = "CT2_test.h5"
config["model_images"] = "CT2_Unet2DBN_muscle_wdscloss.h5"
config["training_model"] = config["model_images"]

config["monitor"] = 'output'
config["data_split"] = 0.0
config["training_split"] = "training_norm_" + str(round(config["data_split"],2)) + '.pkl'
config["validation_split"] = "validation_norm_" + str(round(1-config["data_split"],2)) + '.pkl'

# config['GPU'] = 1
# config['CPU'] = 12
# config['batch_size'] = 4
# config['n_epochs'] = 500
# config['patch_shape'] = None
# config['skip_blank'] = False


def create_validation_split(problem_type,data, training_file, validation_file,data_split=0.9,testing_file=None, valid_test_split=0,overwrite=0):
    """
    Splits the data into the training and validation indices list.
    :param data_file: pytables hdf5 data file
    :param training_file:
    :param validation_file:
    :param data_split:
    :param overwrite:
    :return:
    """
    if overwrite or not os.path.exists(training_file):
        print("Creating validation split...")
        nb_samples = data.shape[0]
        print('Total Samples : ', nb_samples)
        sample_list = list(range(nb_samples))
        if problem_type is 'Classification':
            truth = data.read()
            classes = np.unique(truth).tolist()
            truth = truth.tolist()
            for i in classes:
                print("Number of samples for class ", i, " is : ", truth.count(i) ,'\n')

            x_train,x_valid,y_train,y_valid = train_test_split(sample_list,truth,stratify=truth,test_size=1-data_split)

            if valid_test_split > 0:
                x_valid,x_test,y_valid,y_test = train_test_split(x_valid,y_valid,stratify=y_valid,test_size=1-valid_test_split)
                pickle_dump(x_test,testing_file)
                print('Test Data Split:')
                for i in classes:
                    print("Number of samples for class ", i, " is : ", y_test.count(i) ,'\n')

            print('Train Data Split:')
            for i in classes:
                print("Number of samples for class ", i, " is : ", y_train.count(i) ,'\n')

            print('Valid Data Split:')
            for i in classes:
                print("Number of samples for class ", i, " is : ", y_valid.count(i) ,'\n')

            pickle_dump(x_train, training_file)
            pickle_dump(x_valid, validation_file)
            return x_train, x_valid
        else:
            training_list, validation_list = split_list(sample_list, split=data_split)
            if valid_test_split > 0:
                validation_list,test_list = split_list(validation_list,split=valid_test_split)
                pickle_dump(test_list,testing_file)
            pickle_dump(training_list, training_file)
            pickle_dump(validation_list, validation_file)
            return training_list, validation_list
    else:
        print("Loading previous validation split...")
        return pickle_load(training_file), pickle_load(validation_file)


def split_list(input_list, split=0.8, shuffle_list=True):
    if shuffle_list:
        shuffle(input_list)
    n_training = int(len(input_list) * split)
    training = input_list[:n_training]
    testing = input_list[n_training:]
    return training, testing

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
    training_file = os.path.abspath(os.path.join('datasets',config['training_split']))
    validation_file = os.path.abspath(os.path.join('datasets',config['validation_split']))

    if 'testing_split' in config:
        testing_file = os.path.abspath(os.path.join('datasets',config['testing_split']))

    if data_file.__contains__('/truth'):
        if config["input_type"] is "Both" and data_file.__contains__('/cldata') and data_file.__contains__('/imdata') :
            training_list, validation_list = \
                create_validation_split(
                    problem_type=config["problem_type"],
                    data=data_file.root.truth,
                    training_file=training_file,
                    validation_file=validation_file,
                    data_split=config["data_split"],
                    overwrite=0
                )

        elif config["input_type"] is "Image" and data_file.__contains__('/imdata'):
            training_list, validation_list = \
                create_validation_split(
                    problem_type=config["problem_type"],
                    data=data_file.root.truth,
                    training_file=training_file,
                    validation_file=validation_file,
                    data_split=config["data_split"],
                    overwrite=0
                )

        elif config["input_type"] is "Clinical" and data_file.__contains__('/cldata'):
            training_list, validation_list = \
                create_validation_split(
                    problem_type=config["problem_type"],
                    data=data_file.root.truth,
                    training_file=training_file,
                    validation_file=validation_file,
                    data_split=config["data_split"],
                    overwrite=0
                )
        else:
            print('Input Type: ',input_type)
            print('Clincial data: ', data_file.__contains__('/cldata'))
            print('Image data: ', data_file.__contains__('/imdata'))
            raise Exception("data file does not contain the input group required to train")
    else:
        print('Truth data: ', data_file.__contains__('/truth'))
        raise Exception("data file does not contain the truth group required to train")

if __name__ == "__main__":
    main()