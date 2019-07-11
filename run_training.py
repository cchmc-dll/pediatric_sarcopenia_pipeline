import sys
from argparse import ArgumentParser
import tables
from pyimagesearch.nn.conv.MLP import MLP, MLP10
from pyimagesearch.nn.conv.Resnet3D import Resnet3D
from pyimagesearch.nn.conv.Unet2D import Unet2D,Unet2D_BN
from keras.utils import plot_model
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
from keras.utils.training_utils import multi_gpu_model
from alt_model_checkpoint import AltModelCheckpoint
import matplotlib.pyplot as plt
from pyimagesearch.utils.generator_utils_Elan import *
from keras import Model
from pyimagesearch.callbacks.datagenerator import *
from keras.layers import concatenate
from keras.layers.core import Dense
from sklearn.model_selection import train_test_split
from unet3d.metrics import dice_coefficient_monitor, weighted_dice_coefficient_loss_2D

# Tensorboard specific imports
from time import time
from tensorflow.python.keras.callbacks import TensorBoard


# Define Problem Configuration:
##
# config["input_type"] = "Image"
# config["image_shape"] = (256,256)
# config["input_images"] = "ImageData"
#
# config["overwrite"] = 1
# config["problem_type"] = "Segmentation"
# config["image_masks"] = ["Muscle"] #["Mask"] #["Label"]   # For Image Masks, will serve as label for segmentation problems
# #config["n_channels"] = 1            # All image channels that will be used as input, image_mask can be input for classification problems and output for segmentation problems.
#
# config["normalize"] = False
#
# config["labels"] = (1,)  # the label numbers on the input image
# config["n_labels"] = len(config["labels"])
#
# config["all_modalities"] =  ["CT"] #["Input"]
# config["training_modalities"] = config["all_modalities"]  # change this if you want to only use some of the modalities
# config["n_channels"] = len(config["training_modalities"])
# config["input_shape"] = tuple([config["n_channels"]] + list(config["image_shape"]))

##
# config["data_file"] = "CT_190PTS.h5"
# config["model_images"] = "Unet2DBN_muscle_wdscloss_2.h5"
# config["training_model"] = config["model_images"]
#
# config["monitor"] = 'output'
# config["data_split"] = 0.80
# config["training_split"] = "training_" + str(round(config["data_split"],2)) + '.pkl'
# config["validation_split"] = "validation_" + str(round(1-config["data_split"],2)) + '.pkl'
#
# config['GPU'] = 2
# config['CPU'] = 12
# config['batch_size'] = 4
# config['n_epochs'] = 10
# config['patch_shape'] = None
# config['skip_blank'] = False


"""
USAGE:

python run_training.py \
     --training_model='./arg_model.h5' \
     --data_file='./datasets/CT_190PTS.h5' \
     --data_split=0.8 \
     --training_split='datasets/training_0.8.pkl' \
     --validation_split='datasets/validation_0.2.pkl' \
     --n_epochs=8 \
     --image_masks=Muscle \
     --problem_type=Segmentation \
"""


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


def step_decay(epoch):
    # initialize the base initial learning rate, drop factor, and
    # epochs to drop every
    initAlpha = 1e-3
    factor = 0.5  # 0.75
    dropEvery = 10  # 5

    # compute learning rate for the current epoch
    alpha = initAlpha * (factor ** np.floor((1 + epoch) / dropEvery))

    # return the learning rate
    return float(alpha)


def parse_command_line_arguments():
    parser = ArgumentParser()

    req_group = parser.add_argument_group(title='Required flags')
    req_group.add_argument('--training_model', required=True, help='Location where trained model will be saved')
    req_group.add_argument('--data_file', required=True, help='Source of images to train with')
    req_group.add_argument('--training_split', required=True, help='.pkl file with the training data split')
    req_group.add_argument('--validation_split', required=True, help='.pkl file with the validation data split')
    req_group.add_argument('--data_split', required=True, type=float, default=0.8)
    req_group.add_argument('--n_epochs', required=True, type=int)
    req_group.add_argument('--image_masks', required=True, help='Comma separated list of mask names, ex: Muscle,Bone,Liver')
    req_group.add_argument('--problem_type', required=True, help='Segmentation, Classification, or Regression, default=Segmentation')
    # req_group.add_argument('-o,', '--output_dir', required=True, help='Path to directory where output files will be saved')

    parser.add_argument('--GPU', default=1, type=int, help='Number of GPUs to use, default=1')
    parser.add_argument('--CPU', default=4, type=int, help='Number of CPU cores to use, default=4')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--patch_shape', default=None)
    parser.add_argument('--skip_blank', action='store_true')
    parser.add_argument('--input_type', default='Image')
    parser.add_argument('--image_shape', default=(256, 256))
    parser.add_argument('--monitor', default='output', help='directory where monitor output goes')
    parser.add_argument('--overwrite', default=1, type=int, help='0=false, 1=true')
    parser.add_argument('--show_plots', action='store_true', help='Shows the plots with matplotlib')

    return parser.parse_args()


def build_config_dict(cmdline_args):
    config = vars(cmdline_args)

    config["labels"] = (1,)  # the label numbers on the input image
    config["n_labels"] = len(config["labels"])
    config["all_modalities"] =  ["CT"] #["Input"]
    config["training_modalities"] = config["all_modalities"]  # change this if you want to only use some of the modalities
    config["n_channels"] = len(config["training_modalities"])
    config["input_shape"] = tuple([config["n_channels"]] + list(config["image_shape"]))
    config['image_masks'] = config['image_masks'].split(',')

    return config


def main(overwrite=False):
    args = parse_command_line_arguments()
    config = build_config_dict(args)

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
        data_file = tables.open_file(os.path.abspath(os.path.join(config["data_file"])), mode='r')
    except:
        raise Exception("Error: Could not open data file, check if config[\"data_file\"] is defined \n")

# Step 3: LOAD DATA
    training_file = os.path.abspath(config['training_split'])
    validation_file = os.path.abspath(config['validation_split'])
    if 'testing_split' in config:
        testing_file = os.path.abspath(os.path.join('datasets',config['testing_split']))
    if data_file.__contains__('/truth'):
        if config["input_type"] is "Both" and data_file.__contains__('/cldata') and data_file.__contains__('/imdata') :
            #truth = data_file.root.truth
            #imdata = data_file.root.imdata
            #cldata = data_file.root.cldata
            training_list, validation_list =  create_validation_split(config["problem_type"],data_file.root.truth,training_file, validation_file,config["data_split"],overwrite=0)
        elif config["input_type"] is "Image" and data_file.__contains__('/imdata'):
            #truth = data_file.root.truth
            #imdata = data_file.root.imdata
            training_list, validation_list =  create_validation_split(config["problem_type"],data_file.root.truth,training_file, validation_file,config["data_split"],overwrite=0)
        elif config["input_type"] is "Clinical" and data_file.__contains__('/cldata'):
            #truth = data_file.root.truth
            #cldata = data_file.root.cldata
            training_list, validation_list =  create_validation_split(config["problem_type"],data_file.root.truth,training_file, validation_file,config["data_split"],overwrite=0)
        else:
            print('Input Type: ',input_type)
            print('Clincial data: ', data_file.__contains__('/cldata'))
            print('Image data: ', data_file.__contains__('/imdata'))
            raise Exception("data file does not contain the input group required to train")
    else:
        print('Truth data: ', data_file.__contains__('/truth'))
        raise Exception("data file does not contain the truth group required to train")
    
   
# Step 4: Define Data Generators
    Ngpus = config['GPU']
    Ncpus = config['CPU']
    batch_size = config['batch_size']*Ngpus
    
    config['validation_batch_size'] = 1
    n_epochs = config['n_epochs']
    num_validation_steps = None
    num_training_steps = None
    if input_type is "Both":
            num_validation_patches,all_patches,validation_list_valid = get_number_of_patches(data_file, validation_list, patch_shape = config["patch_shape"],skip_blank=config["skip_blank"],patch_overlap=config["validation_patch_overlap"])
            num_training_patches,all_patches,training_list_valid =     get_number_of_patches(data_file, training_list, patch_shape = config["patch_shape"],skip_blank=config["skip_blank"],patch_overlap=config["validation_patch_overlap"])
            num_validation_steps = get_number_of_steps(num_validation_patches,config["validation_batch_size"])
            num_training_steps =  get_number_of_steps(num_training_patches, batch_size)
       
            training_generator = DataGenerator_3DCL_Classification(data_file, training_list_valid,
                                        batch_size=config['batch_size'],
                                        n_classes=config['n_classes'],
                                        classes = classes,
                                        augment=config['augment'],
                                        augment_flip=config['flip'],
                                        augment_distortion_factor=config['distort'],
                                        skip_blank=False,
                                        permute=config['permute'],reduce=config['reduce'])
            validation_generator = DataGenerator_3DCL_Classification(data_file, validation_list_valid,
                                        batch_size=config['validation_batch_size'],
                                        n_classes=config['n_classes'],
                                        classes = classes,
                                        augment=config['augment'],
                                        augment_flip=config['flip'],
                                        augment_distortion_factor=config['distort'],
                                        skip_blank=False,
                                        permute=config['permute'],reduce=config['reduce'])
    elif input_type is "Image":
            num_validation_patches,all_patches,validation_list_valid = get_number_of_patches(data_file, validation_list, patch_shape = config["patch_shape"],skip_blank=config["skip_blank"])
            num_training_patches,all_patches,training_list_valid =     get_number_of_patches(data_file, training_list, patch_shape = config["patch_shape"],skip_blank=config["skip_blank"])
            num_validation_steps = get_number_of_steps(num_validation_patches,config["validation_batch_size"])
            num_training_steps =  get_number_of_steps(num_training_patches, batch_size)
            
            training_generator = DataGenerator_2D_Segmentation(data_file, training_list_valid,
                                        batch_size=config['batch_size'],
                                        n_labels=config['n_labels'],
                                        labels = config['labels'],
                                        shuffle_index_list=True)
            validation_generator = DataGenerator_2D_Segmentation(data_file, validation_list_valid,
                                        batch_size=config['batch_size'],
                                        n_labels=config['n_labels'],
                                        labels = config['labels'],
                                        shuffle_index_list=True)
    elif input_type is "Clinical":
        validation_list_valid =  validation_list
        num_validation_patches = len(validation_list)
        training_list_valid =  training_list
        num_training_patches = len(training_list_valid)
        num_validation_steps = get_number_of_steps(num_validation_patches,config["validation_batch_size"])
        num_training_steps =  get_number_of_steps(num_training_patches, batch_size)
            
        training_generator = DataGenerator_CL_Classification(data_file, training_list_valid,
                                    batch_size=config['batch_size'],
                                    n_classes=config['n_classes'],
                                    classes = classes)
        validation_generator = DataGenerator_CL_Classification(data_file, validation_list_valid,
                                    batch_size=config['validation_batch_size'],
                                    n_classes=config['n_classes'],
                                    classes = classes)

# Step 5: Load Model
    model1 = None
    if input_type is "Both":
        # create the MLP and CNN models
        mlp = MLP.build(dim=config['CL_features'],num_outputs=8,branch=True)
        cnn = Resnet3D.build_resnet_18(config['input_shape'],num_outputs=8,branch=True)
 
        # create the input to our final set of layers as the *output* of both
        # the MLP and CNN
        combinedInput = concatenate([mlp.output, cnn.output])
 
        # our final FC layer head will have two dense layers, the final one is the fused classification head
        x = Dense(8, activation="relu")(combinedInput)
        x = Dense(4, activation="relu")(x)
        x = Dense(2, activation="softmax")(x)
 
        # our final model will accept categorical/numerical data on the MLP
        # input and images on the CNN input, outputting a single value (the
        # predicted price of the house)
        model1 = Model(inputs=[mlp.input, cnn.input], outputs=x)
        plot_model(model1, to_file="Combined.png", show_shapes=True)
    elif input_type is "Image":
        # create the MLP and CNN models
        model1 = Unet2D_BN.build(config['input_shape'],config["n_labels"])
        # plot_model(model1, to_file="Unet-2D.png", show_shapes=True)
    elif input_type is "Clinical":
        # create the MLP and CNN models
        model1 = MLP.build(dim=config['CL_features'],num_outputs=2,branch=False)
        plot_model(model1, to_file="MLP.png", show_shapes=True)

# Step 6: Train Model
    # Paths for Monitoring
    figPath = os.path.sep.join([config["monitor"], "{}.png".format(os.getpid())])
    jsonPath = None
    
    # OPTIMIZER
    #opt = SGD(lr=1e-4, momentum=0.9) # Continuous Learning Rate Decay
    opt = Adam(lr = 1e-4)
    loss_func =  weighted_dice_coefficient_loss_2D #"binary_crossentropy" #  dice_coefficient_loss 
   # class_weight = {0: 1.,1: 50.}

    ## Make Model MultiGPU
    if Ngpus > 1:
        model = multi_gpu_model(model1, gpus=Ngpus)
        model1.compile(loss=loss_func, optimizer=opt,metrics=["accuracy",dice_coefficient_monitor])
    else:
        model = model1

    model.compile(loss=loss_func, optimizer=opt,metrics=["accuracy",dice_coefficient_monitor])

    # Define Callbacks
    earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=20, verbose=0, mode='auto')
    checkpoint = AltModelCheckpoint(config["training_model"], model1, monitor="val_loss",save_best_only=True, verbose=1)
    #callbacks = [TrainingMonitor(figPath,jsonPath=jsonPath)]
    tensorboard = TensorBoard(log_dir=os.path.join(config['monitor'], str(time())))
    callbacks = [LearningRateScheduler(step_decay),tensorboard,checkpoint,earlystop]

    
    # print Model Summary
    print('Training Model')
    print(model1.summary())
    print('GPU of Training Model')
    print(model.summary())

    # train the network
    print("[INFO] training network...")
    H = model.fit_generator(generator=training_generator,
                        steps_per_epoch=num_training_steps,
                        epochs=n_epochs,
                        validation_data=validation_generator,
                        validation_steps=num_validation_steps,
                        callbacks=callbacks,
                        use_multiprocessing=False, workers=Ncpus)
 
# Step 7: Print Output
  # plot the training + testing loss and accuracy
    Fepochs = len(H.history['loss'])
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, Fepochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, Fepochs), H.history["val_loss"], label="val_loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend()
    figpath_final = 'datasets/' + config["input_type"]+'_loss.png'
    plt.savefig(figpath_final)
    if config['show_plots']:
        plt.show()

    plt.figure()
    plt.plot(np.arange(0, Fepochs), H.history["acc"], label="train_accuracy")
    plt.plot(np.arange(0, Fepochs), H.history["val_acc"], label="val_accuracy")
    plt.title("Training Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend()
    figpath_final = 'datasets/' + config["input_type"]+'_acc.png'
    plt.savefig(figpath_final)
    if config['show_plots']:
        plt.show()

if __name__ == "__main__":
    main()
