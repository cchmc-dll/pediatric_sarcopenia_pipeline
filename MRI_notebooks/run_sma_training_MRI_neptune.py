{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Notebook to Segment Skeletal Muscle Area from MRI images (TIFs)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "###### Load libraries and directories"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "https://app.neptune.ai/chezhia/SMAsegmentation/e/SMAS-2\n",
      "Remember to stop your run once you’ve finished logging your metadata (https://docs.neptune.ai/api-reference/run#stop). It will be stopped automatically only when the notebook kernel/interactive console is terminated.\n"
     ]
    }
   ],
   "source": [
    "import neptune.new as neptune\n",
    "from neptune.new.integrations.tensorflow_keras import NeptuneCallback\n",
    "\n",
    "run = neptune.init(\n",
    "    project=\"chezhia/SMAsegmentation\",\n",
    "    api_token=\"eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlNmVjYjA1ZS1jY2U4LTRiOTctODJiZC00YjExYzk2MzY2N2MifQ==\",\n",
    ")  # your credentials"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "/tf/smipipeline\n"
     ]
    }
   ],
   "source": [
    "import os\n",
    "os.getcwd()\n",
    "os.chdir('/tf/smipipeline')\n",
    "print(os.getcwd())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "from IPython.display import display, HTML"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Using TensorFlow backend.\n"
     ]
    }
   ],
   "source": [
    "# from IPython import get_ipython\n",
    "from tqdm.notebook import tqdm\n",
    "import pickle\n",
    "import pprint\n",
    "pp = pprint.PrettyPrinter(indent=1)\n",
    "\n",
    "# Custom modules for debugging\n",
    "from SliceViewer import ImageSliceViewer3D, ImageSliceViewer3D_1view,ImageSliceViewer3D_2views\n",
    "from investigate import *\n",
    "\n",
    "#pd.set_option(\"display.max_rows\", 10)\n",
    "      \n",
    "import json\n",
    "from run_sma_experiment import find_l3_images,output_images\n",
    "import pprint\n",
    "from L3_finder import *\n",
    "\n",
    "# Custom functions\n",
    "import pickle\n",
    "def save_object(obj, filename):\n",
    "    with open(filename, 'wb') as output:  # Overwrites any existing file.\n",
    "        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)\n",
    "\n",
    "def load_object(filename):        \n",
    "    with open(filename, 'rb') as input:\n",
    "        return pickle.load(input)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "import csv\n",
    "import sys\n",
    "from collections import defaultdict\n",
    "from pathlib import Path\n",
    "import tables"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "No traceback available to show.\n"
     ]
    }
   ],
   "source": [
    "get_ipython().run_line_magic('tb', '')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "/tf/smipipeline\n"
     ]
    }
   ],
   "source": [
    "cwd = os.getcwd()\n",
    "print(cwd)\n",
    "data = '/tf/data'\n",
    "pickles = '/tf/pickles'\n",
    "models = '/tf/models'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{'CPU': 8,\n",
      " 'GPU': 1,\n",
      " 'all_modalities': ['MR'],\n",
      " 'batch_size': 4,\n",
      " 'data_file': '/tf/data/sarcopeniaMR_L3_h5/mri.h5',\n",
      " 'data_split': [0.7, 0.1, 0.2],\n",
      " 'image_masks': ['truth'],\n",
      " 'image_shape': [256, 256],\n",
      " 'input_shape': [1, 256, 256],\n",
      " 'input_type': 'Image',\n",
      " 'labels': ['1'],\n",
      " 'monitor': 'output',\n",
      " 'n_channels': 1,\n",
      " 'n_epochs': 100,\n",
      " 'n_labels': 1,\n",
      " 'output_dir': '/tf/models/muscle/mri/',\n",
      " 'overwrite': 1,\n",
      " 'problem_type': 'Segmentation',\n",
      " 'show_plots': False,\n",
      " 'skip_blank': False,\n",
      " 'testing_split': '/tf/pickles/mri/test_0.2.pkl',\n",
      " 'training_modalities': ['MR'],\n",
      " 'training_model_name': 'ct_mri_invert_retrain.h5',\n",
      " 'training_split': '/tf/pickles/mri/train_0.7.pkl',\n",
      " 'transfer_learning_model': '/tf/models/muscle/cv_final/combined_2020-02-18_dice_fold_0.h5',\n",
      " 'validation_split': '/tf/pickles/mri/validation_0.1.pkl'}\n"
     ]
    }
   ],
   "source": [
    "# Import modules and config file\n",
    "configfile = os.path.join(cwd,'config/mri/sma_mri_retrain.json')\n",
    "with open(configfile, \"r\") as f:\n",
    "        config = json.load(f)\n",
    "pp.pprint(config)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Segment L3 Axial Images and Calculate Muscle Area"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "# debug\n",
    "#with open(f'/tf/smipipeline/config/mri/sma_mri_retrain.json', \"w\") as outfile:\n",
    "#    json.dump(config, outfile)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/usr/local/lib/python3.6/dist-packages/sklearn/externals/joblib/__init__.py:15: DeprecationWarning: sklearn.externals.joblib is deprecated in 0.21 and will be removed in 0.23. Please import this functionality directly from joblib, which can be installed with: pip install joblib. If this warning is raised when loading pickled models, you may need to re-serialize those models with scikit-learn 0.21+.\n",
      "  warnings.warn(msg, category=DeprecationWarning)\n"
     ]
    }
   ],
   "source": [
    "# Import functions from run_training.py\n",
    "# The cells below will also be based on code in run_training.py\n",
    "from run_training import *"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 1: Check if training type is defined\n",
    "try:\n",
    "    input_type = config[\"input_type\"]\n",
    "except:\n",
    "    raise Exception(\"Error: Input type not defined | \\t Set  config[\\\"input_type\\\"] to \\\"Image\\\", \\\"Clinical\\\" or \\\"Both\\\" \\n\")\n",
    "\n",
    "try:\n",
    "    problem_type = config[\"problem_type\"]\n",
    "except:\n",
    "    raise Exception(\"Error: Problem type not defined | \\t Set  config[\\\"problem_type\\\"] to \\\"Classification\\\", \\\"Segmentation\\\" or \\\"Regression\\\" \\n\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "/tf/data/sarcopeniaMR_L3_h5/mri.h5\n"
     ]
    }
   ],
   "source": [
    "print(config['data_file'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 2: Check if the Data File is defined and open it\n",
    "try:\n",
    "    data_file = tables.open_file(config[\"data_file\"], mode='r')\n",
    "except:\n",
    "    raise Exception(\"Error: Could not open data file, check if config[\\\"data_file\\\"] is defined \\n\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Step 3: LOAD DATA\n",
    "training_file = os.path.abspath(config['training_split'])\n",
    "validation_file = os.path.abspath(config['validation_split'])\n",
    "if 'testing_split' in config:\n",
    "    testing_file = os.path.abspath(config['testing_split'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "training pickles:  /tf/pickles/mri/train_0.7.pkl\n",
      "validation pickles:  /tf/pickles/mri/validation_0.1.pkl\n",
      "testing pickles:  /tf/pickles/mri/test_0.2.pkl\n"
     ]
    }
   ],
   "source": [
    "print(\"training pickles: \", training_file)\n",
    "print(\"validation pickles: \", validation_file)\n",
    "print(\"testing pickles: \", testing_file)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.7\n",
      "0.3333333333333333\n"
     ]
    }
   ],
   "source": [
    "# Create train_test_valid splits from data_split configuration which is a ratio list for [train,valid,test].\n",
    "data_split = config['data_split']\n",
    "train_valid_split = data_split[0]\n",
    "print(train_valid_split)\n",
    "valid_test_split = data_split[1]/(data_split[1] + data_split[2] )\n",
    "print(valid_test_split)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Loading previous validation split...\n",
      "No of training samples:  140\n",
      "No of validation samples:  20\n"
     ]
    }
   ],
   "source": [
    "# For imaging data alone\n",
    "if config[\"input_type\"] == \"Image\" and data_file.__contains__('/imdata'):\n",
    "    training_list, validation_list =  create_validation_split(config[\"problem_type\"],\n",
    "                                                              data_file.root.truth,\n",
    "                                                              training_file, \n",
    "                                                              validation_file,\n",
    "                                                              train_valid_split,\n",
    "                                                              testing_file,valid_test_split,\n",
    "                                                              overwrite=0)\n",
    "    print('No of training samples: ', len(training_list))\n",
    "    print('No of validation samples: ', len(validation_list))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "num_training_steps:  35\n",
      "num_validation_steps:  5\n"
     ]
    }
   ],
   "source": [
    "# Step 4: Define Data Generators\n",
    "Ngpus = config['GPU']\n",
    "Ncpus = config['CPU']\n",
    "batch_size = config['batch_size']*Ngpus\n",
    "\n",
    "config['validation_batch_size'] = batch_size\n",
    "n_epochs = config['n_epochs']\n",
    "num_validation_steps = None\n",
    "num_training_steps = None\n",
    "\n",
    "num_validation_patches,all_patches,validation_list_valid = get_number_of_patches(data_file, validation_list)\n",
    "num_training_patches,all_patches,training_list_valid =   get_number_of_patches(data_file, training_list)\n",
    "num_validation_steps = get_number_of_steps(num_validation_patches,config[\"validation_batch_size\"])\n",
    "num_training_steps =  get_number_of_steps(num_training_patches, batch_size)\n",
    "\n",
    "print(\"num_training_steps: \", num_training_steps)\n",
    "print(\"num_validation_steps: \", num_validation_steps)\n",
    "\n",
    "training_generator = DataGenerator_2D_Segmentation(data_file, training_list_valid,\n",
    "                            batch_size=config['batch_size'],\n",
    "                            n_labels=config['n_labels'],\n",
    "                            labels = config['labels'],\n",
    "                            shuffle_index_list=True)\n",
    "validation_generator = DataGenerator_2D_Segmentation(data_file, validation_list_valid,\n",
    "                            batch_size=config['batch_size'],\n",
    "                            n_labels=config['n_labels'],\n",
    "                            labels = config['labels'],\n",
    "                            shuffle_index_list=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Loading transfer learning at: /tf/models/muscle/cv_final/combined_2020-02-18_dice_fold_0.h5\n"
     ]
    }
   ],
   "source": [
    "# Step 5: Load Model\n",
    "from run_sma_experiment import configure_and_load_model # Function to load pre-trained models from CT SMA project\n",
    "model = None\n",
    "model1 = None\n",
    "if 'transfer_learning_model' in config:\n",
    "    print('Loading transfer learning at:',config['transfer_learning_model'])\n",
    "    model1 = configure_and_load_model(config['transfer_learning_model'])\n",
    "else:\n",
    "    if input_type == \"Image\":        \n",
    "        # create the MLP and CNN models\n",
    "        model1 = Unet2D_BN_MOD.build(config['input_shape'],config[\"n_labels\"])\n",
    "        print(\"Fresh model ready\")\n",
    "        # plot_model(model1, to_file=\"Unet-2D.png\", show_shapes=True)\n",
    "        # Step 6: Train Model\n",
    "\n",
    "# Paths for Monitoring\n",
    "figPath = os.path.sep.join([config[\"monitor\"], \"{}.png\".format(os.getpid())])\n",
    "jsonPath = None\n",
    "\n",
    "# OPTIMIZER\n",
    "#opt = SGD(lr=1e-4, momentum=0.9) # Continuous Learning Rate Decay\n",
    "opt = Adam(lr = 1e-3)\n",
    "loss_func = weighted_dice_coefficient_loss_2D #\"binary_crossentropy\" #  #  dice_coefficient_loss\n",
    "# class_weight = {0: 1.,1: 50.}\n",
    "\n",
    "## Make Model MultiGPU\n",
    "if Ngpus > 1:\n",
    "    model = multi_gpu_model(model1, gpus=Ngpus)\n",
    "    model1.compile(loss=loss_func, optimizer=opt,metrics=[\"accuracy\",dice_coefficient_monitor])\n",
    "else:\n",
    "    model = model1\n",
    "\n",
    "\n",
    "model.compile(loss=loss_func, optimizer=opt,metrics=[\"accuracy\",dice_coefficient_monitor])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Current training model is saved at:  /tf/models/muscle/mri/ct_mri_invert_retrain.h5\n"
     ]
    }
   ],
   "source": [
    "# Define Callbacks\n",
    "config[\"training_model\"] = os.path.join(config['output_dir'],config['training_model_name'])\n",
    "print('Current training model is saved at: ', config['training_model'])\n",
    "earlystop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=20, verbose=0, mode='auto')\n",
    "checkpoint = ModelCheckpoint(config[\"training_model\"], monitor=\"val_loss\",save_best_only=True, verbose=1)\n",
    "#callbacks = [TrainingMonitor(figPath,jsonPath=jsonPath)]\n",
    "tensorboard = TensorBoard(log_dir=os.path.join(config['monitor'], str(time())))\n",
    "callbacks = [LearningRateScheduler(step_decay),tensorboard,checkpoint,earlystop]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "GPU of Training Model\n",
      "Model: \"model_1\"\n",
      "__________________________________________________________________________________________________\n",
      "Layer (type)                    Output Shape         Param #     Connected to                     \n",
      "==================================================================================================\n",
      "input_1 (InputLayer)            [(None, 1, 256, 256) 0                                            \n",
      "__________________________________________________________________________________________________\n",
      "conv2d_1 (Conv2D)               (None, 16, 256, 256) 160         input_1[0][0]                    \n",
      "__________________________________________________________________________________________________\n",
      "batch_normalization_1 (BatchNor (None, 16, 256, 256) 1024        conv2d_1[0][0]                   \n",
      "__________________________________________________________________________________________________\n",
      "activation_1 (Activation)       (None, 16, 256, 256) 0           batch_normalization_1[0][0]      \n",
      "__________________________________________________________________________________________________\n",
      "conv2d_2 (Conv2D)               (None, 16, 256, 256) 2320        activation_1[0][0]               \n",
      "__________________________________________________________________________________________________\n",
      "batch_normalization_2 (BatchNor (None, 16, 256, 256) 1024        conv2d_2[0][0]                   \n",
      "__________________________________________________________________________________________________\n",
      "activation_2 (Activation)       (None, 16, 256, 256) 0           batch_normalization_2[0][0]      \n",
      "__________________________________________________________________________________________________\n",
      "max_pooling2d_1 (MaxPooling2D)  (None, 16, 128, 128) 0           activation_2[0][0]               \n",
      "__________________________________________________________________________________________________\n",
      "dropout_1 (Dropout)             (None, 16, 128, 128) 0           max_pooling2d_1[0][0]            \n",
      "__________________________________________________________________________________________________\n",
      "conv2d_3 (Conv2D)               (None, 32, 128, 128) 4640        dropout_1[0][0]                  \n",
      "__________________________________________________________________________________________________\n",
      "batch_normalization_3 (BatchNor (None, 32, 128, 128) 512         conv2d_3[0][0]                   \n",
      "__________________________________________________________________________________________________\n",
      "activation_3 (Activation)       (None, 32, 128, 128) 0           batch_normalization_3[0][0]      \n",
      "__________________________________________________________________________________________________\n",
      "conv2d_4 (Conv2D)               (None, 32, 128, 128) 9248        activation_3[0][0]               \n",
      "__________________________________________________________________________________________________\n",
      "batch_normalization_4 (BatchNor (None, 32, 128, 128) 512         conv2d_4[0][0]                   \n",
      "__________________________________________________________________________________________________\n",
      "activation_4 (Activation)       (None, 32, 128, 128) 0           batch_normalization_4[0][0]      \n",
      "__________________________________________________________________________________________________\n",
      "max_pooling2d_2 (MaxPooling2D)  (None, 32, 64, 64)   0           activation_4[0][0]               \n",
      "__________________________________________________________________________________________________\n",
      "dropout_2 (Dropout)             (None, 32, 64, 64)   0           max_pooling2d_2[0][0]            \n",
      "__________________________________________________________________________________________________\n",
      "conv2d_5 (Conv2D)               (None, 64, 64, 64)   18496       dropout_2[0][0]                  \n",
      "__________________________________________________________________________________________________\n",
      "batch_normalization_5 (BatchNor (None, 64, 64, 64)   256         conv2d_5[0][0]                   \n",
      "__________________________________________________________________________________________________\n",
      "activation_5 (Activation)       (None, 64, 64, 64)   0           batch_normalization_5[0][0]      \n",
      "__________________________________________________________________________________________________\n",
      "conv2d_6 (Conv2D)               (None, 64, 64, 64)   36928       activation_5[0][0]               \n",
      "__________________________________________________________________________________________________\n",
      "batch_normalization_6 (BatchNor (None, 64, 64, 64)   256         conv2d_6[0][0]                   \n",
      "__________________________________________________________________________________________________\n",
      "activation_6 (Activation)       (None, 64, 64, 64)   0           batch_normalization_6[0][0]      \n",
      "__________________________________________________________________________________________________\n",
      "max_pooling2d_3 (MaxPooling2D)  (None, 64, 32, 32)   0           activation_6[0][0]               \n",
      "__________________________________________________________________________________________________\n",
      "dropout_3 (Dropout)             (None, 64, 32, 32)   0           max_pooling2d_3[0][0]            \n",
      "__________________________________________________________________________________________________\n",
      "conv2d_7 (Conv2D)               (None, 128, 32, 32)  73856       dropout_3[0][0]                  \n",
      "__________________________________________________________________________________________________\n",
      "batch_normalization_7 (BatchNor (None, 128, 32, 32)  128         conv2d_7[0][0]                   \n",
      "__________________________________________________________________________________________________\n",
      "activation_7 (Activation)       (None, 128, 32, 32)  0           batch_normalization_7[0][0]      \n",
      "__________________________________________________________________________________________________\n",
      "conv2d_8 (Conv2D)               (None, 128, 32, 32)  147584      activation_7[0][0]               \n",
      "__________________________________________________________________________________________________\n",
      "batch_normalization_8 (BatchNor (None, 128, 32, 32)  128         conv2d_8[0][0]                   \n",
      "__________________________________________________________________________________________________\n",
      "activation_8 (Activation)       (None, 128, 32, 32)  0           batch_normalization_8[0][0]      \n",
      "__________________________________________________________________________________________________\n",
      "max_pooling2d_4 (MaxPooling2D)  (None, 128, 16, 16)  0           activation_8[0][0]               \n",
      "__________________________________________________________________________________________________\n",
      "dropout_4 (Dropout)             (None, 128, 16, 16)  0           max_pooling2d_4[0][0]            \n",
      "__________________________________________________________________________________________________\n",
      "conv2d_9 (Conv2D)               (None, 256, 16, 16)  295168      dropout_4[0][0]                  \n",
      "__________________________________________________________________________________________________\n",
      "batch_normalization_9 (BatchNor (None, 256, 16, 16)  64          conv2d_9[0][0]                   \n",
      "__________________________________________________________________________________________________\n",
      "activation_9 (Activation)       (None, 256, 16, 16)  0           batch_normalization_9[0][0]      \n",
      "__________________________________________________________________________________________________\n",
      "conv2d_10 (Conv2D)              (None, 256, 16, 16)  590080      activation_9[0][0]               \n",
      "__________________________________________________________________________________________________\n",
      "batch_normalization_10 (BatchNo (None, 256, 16, 16)  64          conv2d_10[0][0]                  \n",
      "__________________________________________________________________________________________________\n",
      "activation_10 (Activation)      (None, 256, 16, 16)  0           batch_normalization_10[0][0]     \n",
      "__________________________________________________________________________________________________\n",
      "max_pooling2d_5 (MaxPooling2D)  (None, 256, 8, 8)    0           activation_10[0][0]              \n",
      "__________________________________________________________________________________________________\n",
      "dropout_5 (Dropout)             (None, 256, 8, 8)    0           max_pooling2d_5[0][0]            \n",
      "__________________________________________________________________________________________________\n",
      "conv2d_11 (Conv2D)              (None, 512, 8, 8)    1180160     dropout_5[0][0]                  \n",
      "__________________________________________________________________________________________________\n",
      "batch_normalization_11 (BatchNo (None, 512, 8, 8)    32          conv2d_11[0][0]                  \n",
      "__________________________________________________________________________________________________\n",
      "activation_11 (Activation)      (None, 512, 8, 8)    0           batch_normalization_11[0][0]     \n",
      "__________________________________________________________________________________________________\n",
      "conv2d_12 (Conv2D)              (None, 512, 8, 8)    2359808     activation_11[0][0]              \n",
      "__________________________________________________________________________________________________\n",
      "batch_normalization_12 (BatchNo (None, 512, 8, 8)    32          conv2d_12[0][0]                  \n",
      "__________________________________________________________________________________________________\n",
      "activation_12 (Activation)      (None, 512, 8, 8)    0           batch_normalization_12[0][0]     \n",
      "__________________________________________________________________________________________________\n",
      "conv2d_transpose_1 (Conv2DTrans (None, 256, 16, 16)  1179904     activation_12[0][0]              \n",
      "__________________________________________________________________________________________________\n",
      "concatenate_1 (Concatenate)     (None, 512, 16, 16)  0           conv2d_transpose_1[0][0]         \n",
      "                                                                 activation_10[0][0]              \n",
      "__________________________________________________________________________________________________\n",
      "dropout_6 (Dropout)             (None, 512, 16, 16)  0           concatenate_1[0][0]              \n",
      "__________________________________________________________________________________________________\n",
      "conv2d_13 (Conv2D)              (None, 256, 16, 16)  1179904     dropout_6[0][0]                  \n",
      "__________________________________________________________________________________________________\n",
      "batch_normalization_13 (BatchNo (None, 256, 16, 16)  64          conv2d_13[0][0]                  \n",
      "__________________________________________________________________________________________________\n",
      "activation_13 (Activation)      (None, 256, 16, 16)  0           batch_normalization_13[0][0]     \n",
      "__________________________________________________________________________________________________\n",
      "conv2d_14 (Conv2D)              (None, 256, 16, 16)  590080      activation_13[0][0]              \n",
      "__________________________________________________________________________________________________\n",
      "batch_normalization_14 (BatchNo (None, 256, 16, 16)  64          conv2d_14[0][0]                  \n",
      "__________________________________________________________________________________________________\n",
      "activation_14 (Activation)      (None, 256, 16, 16)  0           batch_normalization_14[0][0]     \n",
      "__________________________________________________________________________________________________\n",
      "conv2d_transpose_2 (Conv2DTrans (None, 128, 32, 32)  295040      activation_14[0][0]              \n",
      "__________________________________________________________________________________________________\n",
      "concatenate_2 (Concatenate)     (None, 256, 32, 32)  0           conv2d_transpose_2[0][0]         \n",
      "                                                                 activation_8[0][0]               \n",
      "__________________________________________________________________________________________________\n",
      "dropout_7 (Dropout)             (None, 256, 32, 32)  0           concatenate_2[0][0]              \n",
      "__________________________________________________________________________________________________\n",
      "conv2d_15 (Conv2D)              (None, 128, 32, 32)  295040      dropout_7[0][0]                  \n",
      "__________________________________________________________________________________________________\n",
      "batch_normalization_15 (BatchNo (None, 128, 32, 32)  128         conv2d_15[0][0]                  \n",
      "__________________________________________________________________________________________________\n",
      "activation_15 (Activation)      (None, 128, 32, 32)  0           batch_normalization_15[0][0]     \n",
      "__________________________________________________________________________________________________\n",
      "conv2d_16 (Conv2D)              (None, 128, 32, 32)  147584      activation_15[0][0]              \n",
      "__________________________________________________________________________________________________\n",
      "batch_normalization_16 (BatchNo (None, 128, 32, 32)  128         conv2d_16[0][0]                  \n",
      "__________________________________________________________________________________________________\n",
      "activation_16 (Activation)      (None, 128, 32, 32)  0           batch_normalization_16[0][0]     \n",
      "__________________________________________________________________________________________________\n",
      "conv2d_transpose_3 (Conv2DTrans (None, 64, 64, 64)   73792       activation_16[0][0]              \n",
      "__________________________________________________________________________________________________\n",
      "concatenate_3 (Concatenate)     (None, 128, 64, 64)  0           conv2d_transpose_3[0][0]         \n",
      "                                                                 activation_6[0][0]               \n",
      "__________________________________________________________________________________________________\n",
      "dropout_8 (Dropout)             (None, 128, 64, 64)  0           concatenate_3[0][0]              \n",
      "__________________________________________________________________________________________________\n",
      "conv2d_17 (Conv2D)              (None, 64, 64, 64)   73792       dropout_8[0][0]                  \n",
      "__________________________________________________________________________________________________\n",
      "batch_normalization_17 (BatchNo (None, 64, 64, 64)   256         conv2d_17[0][0]                  \n",
      "__________________________________________________________________________________________________\n",
      "activation_17 (Activation)      (None, 64, 64, 64)   0           batch_normalization_17[0][0]     \n",
      "__________________________________________________________________________________________________\n",
      "conv2d_18 (Conv2D)              (None, 64, 64, 64)   36928       activation_17[0][0]              \n",
      "__________________________________________________________________________________________________\n",
      "batch_normalization_18 (BatchNo (None, 64, 64, 64)   256         conv2d_18[0][0]                  \n",
      "__________________________________________________________________________________________________\n",
      "activation_18 (Activation)      (None, 64, 64, 64)   0           batch_normalization_18[0][0]     \n",
      "__________________________________________________________________________________________________\n",
      "conv2d_transpose_4 (Conv2DTrans (None, 32, 128, 128) 18464       activation_18[0][0]              \n",
      "__________________________________________________________________________________________________\n",
      "concatenate_4 (Concatenate)     (None, 64, 128, 128) 0           conv2d_transpose_4[0][0]         \n",
      "                                                                 activation_4[0][0]               \n",
      "__________________________________________________________________________________________________\n",
      "dropout_9 (Dropout)             (None, 64, 128, 128) 0           concatenate_4[0][0]              \n",
      "__________________________________________________________________________________________________\n",
      "conv2d_19 (Conv2D)              (None, 32, 128, 128) 18464       dropout_9[0][0]                  \n",
      "__________________________________________________________________________________________________\n",
      "batch_normalization_19 (BatchNo (None, 32, 128, 128) 512         conv2d_19[0][0]                  \n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "__________________________________________________________________________________________________\n",
      "activation_19 (Activation)      (None, 32, 128, 128) 0           batch_normalization_19[0][0]     \n",
      "__________________________________________________________________________________________________\n",
      "conv2d_20 (Conv2D)              (None, 32, 128, 128) 9248        activation_19[0][0]              \n",
      "__________________________________________________________________________________________________\n",
      "batch_normalization_20 (BatchNo (None, 32, 128, 128) 512         conv2d_20[0][0]                  \n",
      "__________________________________________________________________________________________________\n",
      "activation_20 (Activation)      (None, 32, 128, 128) 0           batch_normalization_20[0][0]     \n",
      "__________________________________________________________________________________________________\n",
      "conv2d_transpose_5 (Conv2DTrans (None, 16, 256, 256) 4624        activation_20[0][0]              \n",
      "__________________________________________________________________________________________________\n",
      "concatenate_5 (Concatenate)     (None, 32, 256, 256) 0           conv2d_transpose_5[0][0]         \n",
      "                                                                 activation_2[0][0]               \n",
      "__________________________________________________________________________________________________\n",
      "dropout_10 (Dropout)            (None, 32, 256, 256) 0           concatenate_5[0][0]              \n",
      "__________________________________________________________________________________________________\n",
      "conv2d_21 (Conv2D)              (None, 16, 256, 256) 4624        dropout_10[0][0]                 \n",
      "__________________________________________________________________________________________________\n",
      "batch_normalization_21 (BatchNo (None, 16, 256, 256) 1024        conv2d_21[0][0]                  \n",
      "__________________________________________________________________________________________________\n",
      "activation_21 (Activation)      (None, 16, 256, 256) 0           batch_normalization_21[0][0]     \n",
      "__________________________________________________________________________________________________\n",
      "conv2d_22 (Conv2D)              (None, 16, 256, 256) 2320        activation_21[0][0]              \n",
      "__________________________________________________________________________________________________\n",
      "batch_normalization_22 (BatchNo (None, 16, 256, 256) 1024        conv2d_22[0][0]                  \n",
      "__________________________________________________________________________________________________\n",
      "activation_22 (Activation)      (None, 16, 256, 256) 0           batch_normalization_22[0][0]     \n",
      "__________________________________________________________________________________________________\n",
      "conv2d_23 (Conv2D)              (None, 1, 256, 256)  17          activation_22[0][0]              \n",
      "==================================================================================================\n",
      "Total params: 8,656,273\n",
      "Trainable params: 8,652,273\n",
      "Non-trainable params: 4,000\n",
      "__________________________________________________________________________________________________\n",
      "None\n"
     ]
    }
   ],
   "source": [
    "# print Model Summary\n",
    "print('GPU of Training Model')\n",
    "print(model.summary())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[INFO] training network...\n",
      "Epoch 1/100\n"
     ]
    },
    {
     "ename": "UnknownError",
     "evalue": " Failed to get convolution algorithm. This is probably because cuDNN failed to initialize, so try looking to see if a warning log message was printed above.\n\t [[node model_1/conv2d_1/Conv2D (defined at <ipython-input-22-a37273f9ba35>:10) ]] [Op:__inference_train_function_9779]\n\nFunction call stack:\ntrain_function\n",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mUnknownError\u001b[0m                              Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-22-a37273f9ba35>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      8\u001b[0m                         \u001b[0mvalidation_steps\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mnum_validation_steps\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      9\u001b[0m                         \u001b[0mcallbacks\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mcallbacks\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 10\u001b[0;31m                         use_multiprocessing=False, workers=Ncpus)\n\u001b[0m",
      "\u001b[0;32m/usr/local/lib/python3.6/dist-packages/tensorflow/python/keras/engine/training.py\u001b[0m in \u001b[0;36m_method_wrapper\u001b[0;34m(self, *args, **kwargs)\u001b[0m\n\u001b[1;32m    106\u001b[0m   \u001b[0;32mdef\u001b[0m \u001b[0m_method_wrapper\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    107\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_in_multi_worker_mode\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m  \u001b[0;31m# pylint: disable=protected-access\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 108\u001b[0;31m       \u001b[0;32mreturn\u001b[0m \u001b[0mmethod\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    109\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    110\u001b[0m     \u001b[0;31m# Running inside `run_distribute_coordinator` already.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/usr/local/lib/python3.6/dist-packages/tensorflow/python/keras/engine/training.py\u001b[0m in \u001b[0;36mfit\u001b[0;34m(self, x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_batch_size, validation_freq, max_queue_size, workers, use_multiprocessing)\u001b[0m\n\u001b[1;32m   1096\u001b[0m                 batch_size=batch_size):\n\u001b[1;32m   1097\u001b[0m               \u001b[0mcallbacks\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mon_train_batch_begin\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mstep\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1098\u001b[0;31m               \u001b[0mtmp_logs\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mtrain_function\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0miterator\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1099\u001b[0m               \u001b[0;32mif\u001b[0m \u001b[0mdata_handler\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mshould_sync\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1100\u001b[0m                 \u001b[0mcontext\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0masync_wait\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/usr/local/lib/python3.6/dist-packages/tensorflow/python/eager/def_function.py\u001b[0m in \u001b[0;36m__call__\u001b[0;34m(self, *args, **kwds)\u001b[0m\n\u001b[1;32m    778\u001b[0m       \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    779\u001b[0m         \u001b[0mcompiler\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;34m\"nonXla\"\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 780\u001b[0;31m         \u001b[0mresult\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_call\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwds\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    781\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    782\u001b[0m       \u001b[0mnew_tracing_count\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_get_tracing_count\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/usr/local/lib/python3.6/dist-packages/tensorflow/python/eager/def_function.py\u001b[0m in \u001b[0;36m_call\u001b[0;34m(self, *args, **kwds)\u001b[0m\n\u001b[1;32m    838\u001b[0m         \u001b[0;31m# Lifting succeeded, so variables are initialized and we can run the\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    839\u001b[0m         \u001b[0;31m# stateless function.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 840\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_stateless_fn\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwds\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    841\u001b[0m     \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    842\u001b[0m       \u001b[0mcanon_args\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mcanon_kwds\u001b[0m \u001b[0;34m=\u001b[0m\u001b[0;31m \u001b[0m\u001b[0;31m\\\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/usr/local/lib/python3.6/dist-packages/tensorflow/python/eager/function.py\u001b[0m in \u001b[0;36m__call__\u001b[0;34m(self, *args, **kwargs)\u001b[0m\n\u001b[1;32m   2827\u001b[0m     \u001b[0;32mwith\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_lock\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   2828\u001b[0m       \u001b[0mgraph_function\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mkwargs\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_maybe_define_function\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 2829\u001b[0;31m     \u001b[0;32mreturn\u001b[0m \u001b[0mgraph_function\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_filtered_call\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m  \u001b[0;31m# pylint: disable=protected-access\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   2830\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   2831\u001b[0m   \u001b[0;34m@\u001b[0m\u001b[0mproperty\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/usr/local/lib/python3.6/dist-packages/tensorflow/python/eager/function.py\u001b[0m in \u001b[0;36m_filtered_call\u001b[0;34m(self, args, kwargs, cancellation_manager)\u001b[0m\n\u001b[1;32m   1846\u001b[0m                            resource_variable_ops.BaseResourceVariable))],\n\u001b[1;32m   1847\u001b[0m         \u001b[0mcaptured_inputs\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcaptured_inputs\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1848\u001b[0;31m         cancellation_manager=cancellation_manager)\n\u001b[0m\u001b[1;32m   1849\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1850\u001b[0m   \u001b[0;32mdef\u001b[0m \u001b[0m_call_flat\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mcaptured_inputs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mcancellation_manager\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mNone\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/usr/local/lib/python3.6/dist-packages/tensorflow/python/eager/function.py\u001b[0m in \u001b[0;36m_call_flat\u001b[0;34m(self, args, captured_inputs, cancellation_manager)\u001b[0m\n\u001b[1;32m   1922\u001b[0m       \u001b[0;31m# No tape is watching; skip to running the function.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1923\u001b[0m       return self._build_call_outputs(self._inference_function.call(\n\u001b[0;32m-> 1924\u001b[0;31m           ctx, args, cancellation_manager=cancellation_manager))\n\u001b[0m\u001b[1;32m   1925\u001b[0m     forward_backward = self._select_forward_and_backward_functions(\n\u001b[1;32m   1926\u001b[0m         \u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/usr/local/lib/python3.6/dist-packages/tensorflow/python/eager/function.py\u001b[0m in \u001b[0;36mcall\u001b[0;34m(self, ctx, args, cancellation_manager)\u001b[0m\n\u001b[1;32m    548\u001b[0m               \u001b[0minputs\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    549\u001b[0m               \u001b[0mattrs\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mattrs\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 550\u001b[0;31m               ctx=ctx)\n\u001b[0m\u001b[1;32m    551\u001b[0m         \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    552\u001b[0m           outputs = execute.execute_with_cancellation(\n",
      "\u001b[0;32m/usr/local/lib/python3.6/dist-packages/tensorflow/python/eager/execute.py\u001b[0m in \u001b[0;36mquick_execute\u001b[0;34m(op_name, num_outputs, inputs, attrs, ctx, name)\u001b[0m\n\u001b[1;32m     58\u001b[0m     \u001b[0mctx\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mensure_initialized\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     59\u001b[0m     tensors = pywrap_tfe.TFE_Py_Execute(ctx._handle, device_name, op_name,\n\u001b[0;32m---> 60\u001b[0;31m                                         inputs, attrs, num_outputs)\n\u001b[0m\u001b[1;32m     61\u001b[0m   \u001b[0;32mexcept\u001b[0m \u001b[0mcore\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_NotOkStatusException\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0me\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     62\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0mname\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mUnknownError\u001b[0m:  Failed to get convolution algorithm. This is probably because cuDNN failed to initialize, so try looking to see if a warning log message was printed above.\n\t [[node model_1/conv2d_1/Conv2D (defined at <ipython-input-22-a37273f9ba35>:10) ]] [Op:__inference_train_function_9779]\n\nFunction call stack:\ntrain_function\n"
     ]
    }
   ],
   "source": [
    "# train the network\n",
    "if __name__ == \"__main__\":\n",
    "    print(\"[INFO] training network...\")\n",
    "    H = model.fit(x=training_generator,\n",
    "                        steps_per_epoch=num_training_steps,\n",
    "                        epochs=n_epochs,\n",
    "                        validation_data=validation_generator,\n",
    "                        validation_steps=num_validation_steps,\n",
    "                        callbacks=callbacks,\n",
    "                        use_multiprocessing=False, workers=Ncpus)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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yZIlDb5ntVopKCjAZrNRUFCAxWJpMO23337LhAkTPFG6uLiYo0ePMnjwYKDm+Nd2P6EvILDm79ISCG44uORXVANwpqLlU/mEEN3L73//e9LT09m0aRNbt27ll7/8JZs3b6Zfv34ArFy5EpvNRkVFBTNmzOD6668nODi41j0OHz7Miy++yPLly7nvvvv45JNPmDVrVqPPraysZNGiRbz33ntERUXxwAMP8OabbzJr1iw2btzIl19+iVLK0zO0evVq3n77bXr37t2i7rim6PRusXHjxpGcnMzMmTNJTk5m/PjxDab95ptvuP322z2v/f39KS8vJysri4iICHbv3k2fPn3atbzK34IGUFYCwQ03F0uqXACUOlztWh4hRP0aamF0pNGjR3sCC8Drr7/Oxo0bgZpZr4cPH64TXPr27cuIESMAGDlyJMePH7/oczIzM+nXrx9RUVEA/PznP+fvf/878+bNw2Qy8dBDD9XquRk3bhyLFi3ixhtvZPr06W1S1x/r9AH9mTNnsnv3bh544AH27NnDzJkzgZov1ksvveRJd/r0afLy8hg2bJjnml6v57777mPlypX87ne/48svv2Tu3LntW2BPy6W40WTFZ4PLuSAjhOh5/Pz8PP/eunUrX331FevWrSMpKYkRI0bUu13MhZOV9Ho9LlfLP0MMBgMbNmxgxowZJCUlcccddwDw3HPP8fDDD5OVlcX06dNbtRK/wWe3+R2bKTAwkMcee6zO9aioKE8UBggLC+Pll1+uk27ChAlMmDChXctYi//Z4FJW0mgyabkI0fP4+/tTWlpa73slJSVYrVZ8fX05ePAgO3bsaLPnRkVFcfz4cQ4fPsyAAQN4//33mTRpEmVlZVRUVHD11Vczfvx4LrvsMgCOHDnCmDFjGDNmDFu2bCErK6tOC6q1Oj24dDtnWy5aaQmNzQE713IprXJ3QKGEEF1BcHAw48ePZ+rUqZjN5lozrSZPnsxbb71FXFwcUVFRjBkzps2eazabWbVqFffdd59nQH/u3LkUFhZy9913U1VVhaZpPP744wA89dRTHD58GE3TuPLKKxk+fHibleUcpbVm28turiUnUWrV1bh/PQs1cw66Gbc2mO7p5BNsO1FKbG9/npjatzXF7BDeNhtJ6tP1tUedysvLa3VFdaTW7sXVFblcrjqbccpJlO1E+figzL41s8UaId1iQoieTLrFWkAFWtDKZEBfCNExfv/739fZveTee+/ltttu66QSXZwElxbQBVpxXqTlUiwtFyFEG3nmmWc6uwjNJt1iLaALtDY6W8zl1iitcqFTUOZw43L32GEtIUQPJcGlBVSgpdExl7JqNxoQ5u+DBpRXy4wxIUTPIsGlBXQBjbdciqtqZoz0DjQC0jUmhOh5JLi0gC7QCuWlaO76g0ZJZc31CEtNcJFBfSFETyPBpQV0gRbQNCgrq/f94rMtlT7SchFCNOLcprv1OX78OFOnTu3A0rQtCS4toAKtNf9oYDryuZaKtFyEED2VTEVuAd254NLAoH7xuW6xQJ+aZA4Z0Beio722PYfDBZVtes8BNjP3juvV4PvPPPMMERER3HXXXUDNFvt6vZ6tW7dSVFSE0+nk4Ycf5rrrrmvWcysrK3n00UfZvXs3er2exx9/nCuuuIL09HQefPBBHA4HmqbxyiuvEB4ezn333cepU6dwu9389re/5eabb25NtVtEgksLeIJLA4P6xVUufHQKu19NcCmRbjEheoSbbrqJxx9/3BNc1q1bx9tvv80999xDYGAgZ86c4cYbb+Taa69t9GDEH3vjjTdQSvHFF19w8OBBbr/9dr766iveeust7rnnHm655RYcDgcul4vNmzcTHh7OW2+9BdSce9UZJLi0gAqsOdCsoc0rSxwuLCY9ep3C30dHqXSLCdHhGmthtJcRI0aQl5dHdnY2+fn5WK1WwsLCeOKJJ/j+++9RSpGdnU1ubi5hYWFNvm9KSgrz5s0DYNCgQURGRnLo0CHGjh3LmjVrOHXqFNOnT2fgwIEMGTKEJ598kqeffpr4+HgmTpzYXtVtlIy5tIDOElTzjwbGXIqrXFjMNZu9BZj00nIRoge54YYb2LBhAx9//DE33XQTH3zwAfn5+WzcuJFNmzZht9vrPcelJX7605/yt7/9DbPZzNy5c/n666+Jiori008/ZciQITz//PMkJCS0ybOaS4JLCyg/f9DrGx1zCTSeDS5GvbRchOhBbrrpJj766CM2bNjADTfcQElJCXa7HR8fH7755htOnDjR7HtOmDCBDz/8EKg5SPHkyZNERUVx9OhR+vfvzz333MN1111HWloa2dnZ+Pr6MmvWLBYsWMCePXvauopNIt1iLaCUAr+ARsdcBthqTpMLNOookQF9IXqMSy+9lLKyMsLDw+nVqxe33HILd955J1dffTUjR45k0KBBzb7nnXfeyaOPPsrVV1+NXq8nISEBk8nEunXreP/99zEYDISFhXH//ffzww8/8NRTT6GUwsfHh2effbYdanlxXeI8l9LSUhISEsjNzSU0NJRFixYREBBQJ93atWvZsWMHmqYRExPDvHnzUEpx6NAhXnzxRRwOB7GxsZ7rF9OS81yg5hyKnF/fBr37ov+v/6nz/px/H+DKfoEsmBDO8q9PcuhMFf/vpoEtelZH8bbzQqQ+XZ+c59L1dfvzXBITE4mJiWHNmjXExMSQmJhYJ016ejrp6emsWLGClStXkpmZSWpqKgCvvvoq9913H2vWrCE7O5tdu3a1f6H9A+ttuZzbtPLcmEugUS+LKIUQPU6XCC4pKSnExcUBEBcXV+fcAqjpinI4HDidTqqrq3G5XFitVgoKCqioqCA6OhqlFFdddVW9+dtcQCCU1h3QL3O40KD2mIvDhbvzG4hCiC4oLS2Na665ptafG264obOL1WpdYsylqKgIm80GQFBQEEVFRXXSREdHM3z4cObPn4+maUybNo3IyEgyMzMJCQnxpAsJCeHMmTPtXmblH4h25ECd6+e2frGYzrZcTHrcGlRUu/E36uukF0K0nS7Qy99sQ4cOZdOmTZ1djDbXYcFl6dKlFBYW1rk+e/bsWq+VUvWOl2RnZ3Py5Eleeuklz/3S0tIwGo1NLkNSUhJJSUkALFu2DLvd3pwqeBgMBnxDe1FeVkpISEit8mY5alozkWHB2O02wkNcwGl8/K3YreYWPa8jGAyGFn89uiKpT9fXHnVSSuF2u/Hx8WnT+zaVwdAlfl9vE9XV1RiNRs8v/s3VYV+JJUuWNPjeue4tm81GQUEBFoulTppt27YxePBgzOaaD+jY2FgyMjK46qqryM/P96TLz88nODi43ufEx8cTHx/ved3SwUS73U6FzgDVDvKyTqJM54PG8Zyz4zBVZeTlucBRDsCx7DyM1V03uHjbgLHUp+trjzppmkZlZSXl5eXNWgHfFkwmU5utX+lsmqah0+mIjIys8z1q6oB+lwiz48aNIzk5mZkzZ5KcnMz48ePrpLHb7XzxxRe4XC40TSM1NZXrr78em82Gr68vGRkZDB48mC+//JJp06a1f6EDAmv+Li2BC4LLueONz425nPtbBvWFaH9KKXx9fTvl2d74C0BrAnSXCC4zZ84kISGBzZs3e6YiQ81ioU2bNrFgwQImTZrE3r17Wbx4MQCjR49m3LhxANx777385S9/weFwMHr0aGJjY9u9zMo/EA1qVumHhHqun9sB+cIV+hdeF0KInqBLBJfAwEAee+yxOtejoqKIiooCQKfTMX/+/HrzR0VFsXLlynYtYx0XtlwuUFzlwqhXmPQ1EV9aLkKInqhLTEXulvzPbl5ZVje4BBr1nuZkgLHmSyz7iwkhehIJLi3VSMvlXJcYgI9eh9mgZH8xIUSPIsGlpfzPbk/zo52RS6pcBJpqr2cJMOplfzEhRI8iwaWFlMEHzL71t1x+FFwCTbIFjBCiZ5Hg0hr17C9WUuX0DOKfI9vuCyF6GgkurRFgQbug5eJya5Q63LXGXOBct5gEFyFEzyHBpTV+1HI5t2ll3W4xOepYCNGzSHBpBfWjnZHPrc63mGovHzo3oN8dN9UTQoiWkODSGj9quXi2fvlxy8Wox+nWqHJJcBFC9AwSXFojIBDKy9BcNUHFs/XLj6ciyxYwQogeRoJLa5xdpU95KXBht1jdlgvIFjBCiJ5Dgktr/GiVfkPBJcB0dgsYabkIIXoICS6toPzPBpezq/Q9m1Yaan9ZpeUihOhpJLi0xo9aLvVt/QLnx1xKZQsYIUQPIcGlNc62XM7tjFzf1i9wvuUi3WJCiJ5CgktrBJwd0C9tPLiYDDqMeiXdYkKIHkOCS2uYfUGv94y5lFQ56+0Wg7MLKaXlIoToISS4tIJSqqZr7IIxl/paLlDTNSYtFyFET9HpxxyXlpaSkJBAbm4uoaGhLFq0iICAgDrp1q5dy44dO9A0jZiYGObNm4fD4WDVqlXk5OSg0+kYO3Ysd9xxR8dWwD8Qrazk/KaVDbVcTDo500UI0WN0esslMTGRmJgY1qxZQ0xMDImJiXXSpKenk56ezooVK1i5ciWZmZmkpqYCcOONN7J69Wqef/550tPT2blzZ8dWIKCm5VJ6dtPKxrrFZPNKIURP0enBJSUlhbi4OADi4uJISUmpk0YphcPhwOl0Ul1djcvlwmq1YjKZGDFiBAAGg4EBAwaQn5/foeXH3wJlJRds/VJ/YzDQJGMuQoieo9O7xYqKirDZbAAEBQVRVFRUJ010dDTDhw9n/vz5aJrGtGnTiIyMrJWmrKyM//znP1x//fUNPispKYmkpCQAli1bht1ub1GZDQaDJ2+RPRTH0YMoc01XXmSoDbvdVidPqLWE0qMlLX5me7uwTt5A6tP1eVudvK0+0Lo6dUhwWbp0KYWFhXWuz549u9ZrpVTNIPmPZGdnc/LkSV566SXP/dLS0hg6dCgALpeLF154genTp9OrV68GyxEfH098fLzndV5eXovqY7fbPXndeh+04kKO5ZxtMVWVkZdXt4VicDmocro5mX26zgr+ruDCOnkDqU/X52118rb6QP11ioiIaFLeDgkuS5YsafA9q9VKQUEBNpuNgoICLBZLnTTbtm1j8ODBmM1mAGJjY8nIyPAEl5dffpnw8HBmzJjRPhVoTEAguJyUlFUCjYy5nN1frNTh6pLBRQgh2lKnf8qNGzeO5ORkAJKTkxk/fnydNHa7nbS0NFwuF06nk9TUVPr06QPAu+++S3l5OXfddVdHFvu8s6v0S0oqgLqbVp5zfn8xmTEmhPB+nT7mMnPmTBISEti8ebNnKjJAZmYmmzZtYsGCBUyaNIm9e/eyePFiAEaPHs24cePIz8/ngw8+oE+fPjzyyCMATJs2jauvvrrDyq8CLGhAcXllvZtWnuPZX0wG9YUQPUCnB5fAwEAee+yxOtejoqKIiooCQKfTMX/+/DppQkJC+Oc//9nuZWzU2ZZLcUU1FpO5wWSe/cVkIaUQogfo9G6xbu/szsjFlfXviOxJJtvuCyF6EAkurXU2uJRUN7w6H+TAMCFEzyLBpbX8zrZcnKrR4OJr0KFXMqAvhOgZJLi0kjIYwNePEk3faHBRShEgq/SFED2EBJc24PK3UKYZGh1zAdkZWQjRc0hwaQOlgXY0pRrcV+ycAKNeZosJIXoECS5toCQwBGh4df45gSZdvetcXtmewyvbc9qlbEII0Rk6fZ2LNyjxq9mosrExF6hpuRwtrKp17dCZSjakF9DHYmy38gkhREeT4NJMu7PLCCjTY9c7sZhrvnwlvlbQmhBcTHpKqmrPFvvH7lwAiiqd7VNgIYToBBJcmumfe/PZk3McALufgYHBZlz6SHBCgEFrNG+gUU+F043TrWHQKdLzKkg5WUaQWU9hpYtql4aPvu6u0EII0d1IcGmmR37Sh3yXiZ1Hcjh0popDBZWcdFrwdVZizT4ClqEN5r1wlX6Q2cDaXblYzXp+OiyYv+3IpbjKSYifTwfVRAgh2o8El2YKNOkZYA/iEr/z3VjlRUVUPboAn+CbIbrh4BJ4weaVRwur2J1Tzj1jwwj1rwkoRZUuCS5CCK8gs8XagJ/ViiWiN1rqrkbTBRjPbwGzdlcuIX4Gpg0OIuhs0CmSBZZCCC8hwaWNqGGj4HA6WmV5g2nOtVy2HC4mI7+S2TF2jHod1rMTA2RQXwjhLSS4tBE1dDS4XJCxr8E058ZcPj9YSHiAD1MHWgGwms+2XCql5SKE8A4SXNrKoKHgY0RL+6HBJOfOdNGA20faMehqZob5+egw6BSF0nIRQngJGdBvI8rHCIOGNhpc/Iw6dAoiLUZ+0t9yPq9SWM16abkIIbxGlwgupaWlJCQkkJub6znqOCAgoE66tWvXsmPHDjRNIyYmhnnz5qHU+XUhzz33HKdPn2blypUdWXwPNXQ02gd/RysqQFltdd7XKcW8MWEMC/VDr6u9niXIrJcxFyGE1+gS3WKJiYnExMSwZs0aYmJiSExMrJMmPT2d9PR0VqxYwcqVK8nMzCQ1NdXz/vfff4/Z3PAxwx1BDRsN0Gjr5aYhwQwKqVtOq8kgs8WEEF6jSwSXlJQU4uLiAIiLiyMlJaVOGqUUDocDp9NJdXU1LpcLq7VmQLyyspL169cza9asDi13HX0HgH8gNBJcGmKVlosQwot0iW6xoqIibLaabqSgoCCKiorqpImOjmb48OHMnz8fTdOYNm0akZGRALz77rvceOONGI2Nb/6YlJREUlISAMuWLcNut7eovAaDocG8haPGU52+h5CQkFpddhcTbith6/HSZudrK43VqTuS+nR93lYnb6sPtK5OHRZcli5dSmFhYZ3rs2fPrvVaKVXvh2t2djYnT57kpZde8twvLS0NX19fcnJyuOuuuzh9+nSjZYiPjyc+Pt7zOi8vryVVwW63N5jXHTUEbetm8vb+gOod2eR7GjUHVU43J7Jz8fXp+AZlY3XqjqQ+XZ+31cnb6gP11ykiIqJJeTssuCxZsqTB96xWKwUFBdhsNgoKCrBYLHXSbNu2jcGDB3vGVWJjY8nIyMDX15dDhw6xcOFCXC4XRUVFPPHEEzzxxBPtVZVGqaGj0QAtbVezgkvQBQspfX1k+30hRPfWJcZcxo0bR3JyMgDJycmMHz++Thq73U5aWhoulwun00lqaip9+vTh2muv5eWXX+bFF1/kySefJCIiotMCC4AKDQd7r0YH9etjPbt6v1CmIwshvECXCC4zZ85k9+7dPPDAA+zZs4eZM2cCkJmZ6ekGmzRpEr169WLx4sX87ne/o3///owbN64zi90gNXQUpO9BczU9UMgWMEIIb9IlBvQDAwN57LHH6lyPiooiKioKAJ1Ox/z58xu9T1hYWKetcall6Gj46nM4cgCihjQpi2cLGJmOLITwAl2i5eJt1JCRoFSzusbOBRfZAkYI4Q0kuLQDFWiBvgObFVyMeh1+PjrZAkYI4RUkuLQTNXQUZO5Hq6psch5ZSCmE8BZNDi7r16/nyJEjAGRkZPBf//VfLFy4kIyMjPYqW7eBIR6fAAAgAElEQVSmho0ClxMy9jY5j9VkkJaLEMIrNDm4bNiwgbCwMADeeecdbrjhBmbNmsUbb7zRXmXr3gYPB6MJbff2JmeRnZGFEN6iycGlvLwcPz8/KioqOHLkCNOnT2fq1KlkZWW1Z/m6LeVjhKGj0PZsR9O0JuUJMhsorJJuMSFE99fk4BISEkJ6ejrffPMNQ4cORafTUV5ejk4nwzYNUSPHQf5pyDrepPRWs56SKhcud9OCkRBCdFVNXucyZ84cVq1ahcFg4KGHHgJgx44dDBo0qN0K192pEeNqtoLZk4Lq0++i6a1mPW4NSh0uz6JKIYTojpr8CTZmzBhefvnlWtcmTZrEpEmT2rxQ3kIF26HvALTdKTDt4scBWE3nVulLcBFCdG9N7tM6ceKEZ1fjyspK/vnPf/Lhhx/iasYWJz2RihlfMyW5rPSiaWUhpRDCWzQ5uLzwwguUl5cD8Oabb5KWlsaBAwd45ZVX2q1w3kCNHAduN9q+HRdNe35nZAnYQojurcl9L6dPnyYiIgJN09i2bRurVq3CaDTym9/8pj3L1/0NGAwBFtidAhOuajTp+f3FpOUihOjemhxcjEYjFRUVnDhxArvdjsViweVyUV1d3Z7l6/aUTo8aMQZt73/Q3C6UTt9g2gCjHp2SlosQovtrcnC54oorePLJJ6moqGDatGkAHD582LOwUjRi5Hj47v/gUAYMGtpgMr1OEWiShZRCiO6vycHlrrvu4ocffkCv1zNixAig5kjiO++8s90K5y3U8Fg0nQ5tz3ZUI8EFIMhkkAF9IUS316wVkKNGjSI8PJyMjAzy8vKIioryBBrRMOUXAIOG1kxJvgjZAkYI4Q2a3HIpKChg9erVHDhwgICAAEpKSoiOjua3v/0twcHB7VlGr6BixqG9/3e0M7mo4NAG01nNeg6eafpOykII0RU1ueXy6quv0r9/f15//XVeeeUV/va3v3HJJZfw6quvtmf5vIYaOR4Abc9/Gk1nNcvOyEKI7q/JLZf09HQefPBBDIaaLGazmTlz5rBgwYJWFaC0tJSEhARyc3MJDQ1l0aJFBAQE1Em3du1aduzYgaZpxMTEMG/ePJRSOJ1O/vrXv5KamopSitmzZ3fNXQN694WQMLQ92yFuWoPJrGY95dVuHC43Rr3s2yaE6J6a/Onl7+/PiRMnal3LysrCz8+vVQVITEwkJiaGNWvWEBMTQ2JiYp006enppKens2LFClauXElmZiapqakAfPDBB1itVl544QVWrVrFsGHDWlWe9qKUqllQmbYLzVHVYDpZSCmE8AZNDi433XQTS5cu5e233+bzzz/n7bff5qmnnuLmm29uVQFSUlKIi4sDIC4ujpSUuoPeSikcDgdOp5Pq6mpcLhdWqxWALVu2MHPmzJrK6HRYLJZWlac9qZjx4HBAesMHiFlNZxdSSnARQnRjTe4Wi4+PJzw8nK+//ppjx45hs9l44IEHPC2IlioqKsJmswEQFBREUVFRnTTR0dEMHz6c+fPno2ka06ZNIzIykrKyMgDee+89UlNT6dWrF3fffTdBQUH1PispKYmkpCQAli1bht1ub1GZDQZDi/JqV0wm99Xl6D//ENuVU1E+PnXS9K82AifRTH7Y7R03UaKldeqqpD5dn7fVydvqA62rU7O23h0xYkStqcfV1dU89dRT3HbbbY3mW7p0qWfTywvNnj271mulFEqpOumys7M5efIkL730kud+aWlp9OnTh/z8fC699FLuvPNO1q9fz1tvvcX9999fbzni4+OJj4/3vM7Ly2u03A2x2+0tzsvchVS/spzcPz2NmruwTn21SgcAx08XMDjA3bJntECr6tQFSX26Pm+rk7fVB+qvU0RERJPydsi+7kuWLGnwPavVSkFBATabjYKCgnq7tbZt28bgwYMxm80AxMbGkpGRwZAhQzCZTEyYMAGoOQJg8+bN7VOJNqIb/xPcxw+jbfw39B2AmjKj1vuyM7IQwht0+nSkcePGkZycDEBycjLjx4+vk8Zut5OWlobL5cLpdJKamkqfPn1QSjF27FhP19zevXuJjIzs0PK3hJo5B0aOR3v3VbT9u2u952vQYdQrGXMRQnRrF2257N3b8OCz09n6365nzpxJQkICmzdv9kxFBsjMzGTTpk0sWLCASZMmsXfvXhYvXgzA6NGjGTduHAB33HEHf/7zn3njjTewWCz8+te/bnWZ2pvS6dDd+xDuZ3+H++Xn0P1+JSo0vOY9pbCa9BRJy0UI0Y0pTdMaPbB94cKFF73Jiy++2GYF6khZWVktytdWfataThbuZx4Cmx3d/zyHMtdM635o4xEsJj2PT+3b6mc0lbf1F0t9uj5vq5O31QfaecyluwaO7kD1ikB338O4V/8v7iW/BksQGE1YQq+lUO+H+99foGbdWe8kByGE6Mo6fcylp1PDYtHd9zvU4OFgs4OPEauzjCLNgPbZB3D8UGcXUQghmq1DZouJxqmxV6DGXuF5bdt5mqL9Z9CUDm3nd6h+UZ1YOiGEaD5puXRBVrMepxvKo0ei7fyus4sjhBDNJsGlC7KaahqUxcMnwcmjaKdPdXKJhBCieSS4dEHnFlIWR40EQNslrRchRPciwaUL8uyM7GuFyAFoO7/v5BIJIUTzSHDpgs61XIoqnajYSZCZhlZc0MmlEkKIppPg0gVZTOfPdFGxk0DT0H6oexSBEEJ0VRJcuiAfvcLfqKvZAibykpoTLJs5a6yi2k15texPJoToHBJcuiiryUBhpavmGILYy2pOsKwsb1Jep1vj95uO8nTyyXYupRBC1E+CSxcVZNZTVFXT8lCxE8HphL07mpT347QzHCqoYn9uOQ5Xx50JI4QQ50hw6aKsZsP5nZEHDYUAS5O6xk6VOHhnTx4hvgacbsjMr2znkgohRF0SXLqoILPec6aL0ulRoyag7dmO5qxuMI+maby0LRu9Uvxxcs25Nml5FR1SXiGEuJAEly7KatZTUuWiylnTraViJ0FFOaQ3fL5O8pFidmWXM3d0KAODzfQO9GF/rgQXIUTHk+DSRcX08kejJmAAMHQUmMxoO7+tN31xpZO//uc0l9p9mR4dBMCldl/251VwkSN7hBCizUlw6aKGh/ky0Gbio7QzaJqGMppg+Bi0XdvQ3HUH6V/fcZoyh4uFE8PRnT3/ZYjdl6JKFzmlDXelCSFEe+gSW+6XlpaSkJBAbm6u56jjgICAOunWrl3Ljh070DSNmJgY5s2bh1KKr7/+mg8//BClFDabjfvvvx+LxdIJNWk7SiluHhpMwtZT7DxVxpiIAFTsRLQdW+HQfhg0zJN216kythwu5ufDQ+gfZPJcHxrqC8D+vArCA40dXgchRM/VJVouiYmJxMTEsGbNGmJiYkhMTKyTJj09nfT0dFasWMHKlSvJzMwkNTUVl8vFG2+8weOPP86KFSvo378/n376aSfUou1d0c+CzdfAR/trtn5RoyeC2Rft/zZ60lQ63fy/bdlEBPpwa0xIrfx9rSZ8DToZdxFCdLguEVxSUlKIi4sDIC4ujpSUuludKKVwOBw4nU6qq6txuVxYrVY0TUPTNKqqqtA0jfLycoKDgzu6Cu3CR6+YER3ErlNlHC2sQpn9UFfEo23/Bq3wDABv7colu7SaX08Mx6iv/e3U6xSX2s3slxljQogO1iW6xYqKirDZbAAEBQVRVFRUJ010dDTDhw9n/vz5aJrGtGnTiIysmW77q1/9isWLF2Mymejduzf33ntvvc9JSkoiKSkJgGXLlmG321tUXoPB0OK8zfWLiVb+te8Mnx8p59H4PjhnzSV/83p8t3/JgSt/zvr0AmaN6s2U4f3rzR/br4y/pxzH1xKEv7Hhb3dH1qkjSH26Pm+rk7fVB1pXpw4LLkuXLqWwsLDO9dmzZ9d6rZRCnR2QvlB2djYnT57kpZde8twvLS2NwYMH8/nnn/Pcc8/Rq1cvXn/9dT788ENmzZpV5x7x8fHEx8d7Xufl5bWoLna7vcV5W2LKAAufpZ3m50MCCTKbYcRY8j79mKWlIwgP8OHWIYENlqefv4Zbg+8yTjIq3L/BZ3R0ndqb1Kfr87Y6eVt9oP46RURENClvhwWXJUuWNPie1WqloKAAm81GQUFBvYPx27ZtY/DgwZjNZgBiY2PJyMjAx8cHgPDwcAAuu+wyPvroo3aoQee5cYiNTw8U8mlGIbNH2tFdfSNrN+zgdFk1T1/TH7Oh4d7NaLsvCtifW9FocBFCiLbUJcZcxo0bR3JyMgDJycmMHz++Thq73U5aWhoulwun00lqaip9+vQhODiYEydOUFxcsx5k9+7d9OnTp0PL394iLSbGRfjzSUYBDpebPSHRbOxzBdcX7WXY2RlhDQkw6ulnNcmgvhCiQ3WJMZeZM2eSkJDA5s2bPVORATIzM9m0aRMLFixg0qRJ7N27l8WLFwMwevRoxo0bB8DPfvYzHn/8cfR6PXa7nYULF3ZaXdrLzUODWfLFcT47UMjH+wvora9mzg/vwqEYiBrSaN5LQ818c7QEt6Z51sAIIUR76hLBJTAwkMcee6zO9aioKKKiogDQ6XTMnz+/3vzXXnst1157bbuWsbPF9PLjkiATf/3PaQCeiYvEtNWI9sU61EWCyxC7L58fLOJEsYN+VlOjaYUQoi10iW4xcXHnFlVqwE1DbAyLtKGujEfbsRWtIL/RvENC/QCka0wI0WEkuHQjkwdYeHxKJHNHhwGgpswAt7vWosr6RAT6EGjSS3ARQnQYCS7diE4pxkQE4KOvGTdRoeEwcjzal5+iVTsazKeUYsjZTSyFEKIjSHDp5nRX3wilxWhbNjSabojdl5PFDorPnm4phBDtSYJLdzdkJIwYg/avv+He8M8Gt9cfcnbKcoa0XoQQHUCCSzenlEK38A+oSZPREteivfUimqtu62RwiBmdgjQZdxFCdIAuMRVZtI4y+MDdiyA4DO2Tf6IV5KO772GU+fwCS5NBx0CbbGIphOgY0nLxEkopdD+dg5r7a0jdiXv579GKCmqlGRLqy4G8CsocMu4ihGhfEly8jO6qaeh+80fIOYl79eNoTqfnvakDrThcGu/u8a7N9YQQXY8EFy+kYsahu3sRnDiC9sU6z/WoYDPXDgpifXoBxwqrOrGEQghvJ8HFW8VOglET0Na9g5af67k8Z5QdPx8dr2zPaXBmWXNUuzSKK50XTyiE6FEkuHgppRS62+eDpuF+91XPdYvZwJxRoezJKeebYyUturfTrbEjq5QXvj3Fne8f4FcfZZJbVt1WRRdCeAGZLebFVEgY6sbZaO//HW3X96jREwG4dlAQnx0s5PUdpxkbEYCvT93fMfLLq8krd1LldFPl1Khyual0utmfW8F3x0socbjx99ExPjKArcdKeGtXLg9e0bRDhIQQ3k+Ci5dT8TejfbsF9zuvoBs6CmUyo9cp7hvXi//ZdIx/78tn7uhQT/pSh4t3d+exIaMAdz29ZmaDjomRAVzZP5DY3v746HXY/XL59758brjURrS98fNlhBA9gwQXL6cMBnR3/Bfu5Y+irX8PNetOAIaG+TFlgIXEtHymDrQSHKKx6WAhb+3KpbjKxbWDgpgYGYDJoMNkUJgMOsx6HUG+eoz62i2dWcOD2ZRZ0xJ69pp+9R5TLYToWSS49AAqejjqini0TYlokyaj+vQH4M7YML47XsqfvjuF9v1p9p8uZWioL0+M68XAYHOT7+/no2fOqFBe/D6brcdLuKJf3WOqhRA9iwzo9xBq1l3g64f7r6twf5+MdiYPm6+B20faScutILfMwaLLe/PsNf2aFVjOuXqglf5WE2/uzKXa5W77CgghuhVpufQQKtCCbs5C3H9fg/baSjQAey+uHzyc8L7jmXzzNCrKWzZ7DECvU8wbG8YTm4+zIaOAmUND2qzsQojup9ODS2lpKQkJCeTm5hIaGsqiRYsICAiok27t2rXs3LkTgFmzZnH55ZcDcPr0aVavXk1JSQkDBw7k/vvvx2Do9Gp1SWrs5ehiJ8LxI2gH9qEd2Iduz3bGfbsZR1oS2r0Pofzqfu2bKra3P2Mj/PnnnnymDrBiMcv3QYieqtO7xRITE4mJiWHNmjXExMSQmJhYJ82OHTs4fPgwzz//PE8//TTr1q2jvLwcqAk6M2bM4E9/+hP+/v5s3ry5o6vQrSidHtU/Cl38Tej/61F0q95C3fVbHGk/4F72CFpudqvuf1dsGBVON+/ubfzoZSGEd+v04JKSkkJcXBwAcXFxpKSk1Elz4sQJhg4dil6vx2w2069fP3bt2oWmaezbt49JkyYBMHny5Hrzi4YppdBdcTW2x1dDUQHuZ3+Hlrm/xffrF2Ti2kFBbMwoYPOhojbZBUAI0f10er9FUVERNpsNgKCgIIqKiuqk6d+/P//+97+58cYbqaqqYt++fURGRlJSUoKfnx96vR6A4OBgzpw50+CzkpKSSEpKAmDZsmXY7fYWldlgMLQ4b1dlCA9H9/yrFD61GNfKP2L97RLMV1zdonvdP8XKsZJUXvj2FFuOlLFoykCiQ2t3t5VVOVmfmsP7P5zC6dK4bmgo04f2op+tbdbJeNv3yNvqA95XJ2+rD7SuTh0SXJYuXUphYWGd67Nnz671WilV7xqJUaNGkZmZyR//+EcsFgvR0dHodM1vdMXHxxMfH+95nZfXst2B7XZ7i/N2VXa7nUKTP9rDz8FfnqFoxRKK0/ehbpiNauRrre36Dm3fTtSM21BBwZ7rT0+NYPOhIt7cmcs97+ziukFB3DEqlPJqF+vTC9h0sIgKp5uhob74+ehZu/0Eb6acYGioL1MHWrmyfyB+PvpW1cebvkfeVh/wvjp5W32g/jpFRDRtJ44OCS5Llixp8D2r1UpBQQE2m42CggIslvrXSNxyyy3ccsstALzwwgv07t2bwMBAysvLcblc6PV6zpw5Q3BwcL35RdOoQAu6B5fWnGi57l20wxno7nkQFVD7+6JVO9D+9Te0LRtqXm/7EnXbvajLptZ0tSlFfFQQkyID+ceePDZmFJB8pJhKpxsFXNHPwk1DbQwOqWmp5JdXk3y4mC8OFfHi99m8/UMuq68fgM230xvXQogW0D/xxBNPdGYB8vLyOHXqFEOGDOGzzz4jNDSUkSNH1krjdrspLS3FZDJx9OhRPv/8c+bOnYtOpyMzMxOAfv368f777zNs2DAGDRrUpGeXlLRs6q2fn59nQoG3uLBOSq+H0RMhKBj+75OawDF4mKdlouVk4X7hCfhhGyr+ZnS//A3a4QzYvB7tyEHU4OEoXz8AjAYdYyMCmBQZQEGlk7ERATx0ZQTxUUGE+Pmcf76PnqFhflwfHURML382HSwip7SaK/o3f0HmzlNlFFWD1dA11tu43BovbcvhZEkVA2xmDLrm72Dg7T9z3sDb6gP11ykwMLBJeZXWySOuJSUlJCQkkJeXV2sqcmZmJps2bWLBggU4HA4eeeQRoKayv/rVr7jkkksAyMnJYfXq1ZSWljJgwADuv/9+fHx8GnnieVlZWS0qc09p/gJohw/gfmkZFBegbr8PTGa0t/4Cej26eb/1bIapud1oWzagffAm6PWon9+NuvKaFm8F86+9eaz9IY/fX9WHiX2b9sPs1jTe2Z3HP/fmY9Qrnrmmn6dl1Jm+O17Cs1+eBCDY18BtMSHERwU1K8j0pJ+57srb6gOt6xbr9ODSmSS4nNdYnbTSYtyvrYR9NeuMiBqC7le/Q4WE1k17+hTuN/8M6Xtg6Ch0cxeiQsObXR6nW2Pxp0coqnTx5xsG4G9sfPylotrN6m+z+O54KVMGWEjLq8LpcrFy+iUEdfJ6mz9sOsrpsmp+M6k3//ghj/15FYQH+HD7SDs/6W9B34Qg09N+5rojb6sPtC64dHq3WGeSbrHzGquTMppQE64CkxkVPQLdL+9HBdTfmlD+gahJU2q61L7djLZlPfgYYcBglGr6JAydUkQFm1mXXkBJlZvxkQ0v7swpdfD45uOk5lZw99gwfjk6lMuje/P+7lOk51YQN8CKroktKIfLzZrvTnGiqIqBNjM++tZtwnnoTCVv/ZDHrTEhTB5gJT7KyuAQX9JyK9h4oJAjhVVc2T/woi28nvYz1x15W32gdd1iElxaoKf8EF1IKR1q0DBU9IhGZ4/VpFWoSwahJk5Gyz5RMxazdwdq4KUoS1CTyxTi50NFtZsNGQWM6OVLrwBjnTT7csp5bPNxSh0uHr2qD5MHWFFK0S80CH+q+Ti9gAqnmzERTdt54NXtOSRlFrE7p5xNmYXodYoBNlOTWhf1eXPXaU6VOFh0eQRGvQ6lFBEWI9cOCkKnFJ8eKKSPxUT/IFOj9+mJP3PdjbfVB1oXXGQqjmg3KtiObuEf0LZ/jfbOK7ifWoSaNBl15TUQNbRJ4zG/GGnnu+MlvPh9Ni9cPwCTQYdb09iRVca6/WfYlV1ORKCRP0zuQ6Sl9gf0lIFWMs9Usm5/AVE2M1MGWht91jfHitl4oJCZQ4O5vF8ga3/I5a//OU1i2hluG2FneJgvRwqrOFJQxZHCSg4XVGHQKZ6/rj/WerreCiqcfHmkhOsGWQn4UbeeTil+PjyEbSdKee0/OYzp7U+AqeVTr4XoaqTl0gI95TeUtqCUQvXpj7oiHsrL0LZ9jZb8KVrKV+CoAns4ytzwoLtBp+gfZOLj/TUtkFMlDlZvPcWGjEIcLo1bhgfzm4nh2P1qT+I4V5+R4f6kni7n04OFjI0IILiBqc05pQ6e3HKCATYTiy6PICzAh6kDrQwP8+XgmUo+PVDIJxmFbD1Wwv68CvRKMSjEzIH8So4WVnHVJZY6wTIxLZ/dOeU8eHkEgfUEDp1SDA452/XncDEhsuHfCOVnruvztvqAtFxEN6ACLKg5v0b72Ty0/3yD9nVSzfHLH74Fw8egJlyFGj2x3kAzMtyfa6KsrE8vAOBSu5mHRoZyeb/Ai864MugUv/tJHx7ceISl/3eChRPC64zfVLs0ln+dhQIWXxlRa5xlZLg/z/XyY1d2OWfKqxlgM9PXasTn7IFpG9ILeGV7DhsyCrjh0vNrrBwuNxsPFDK+jz8RlrrdeecMDDZz05BgEtPOMHmAleFhfhf9Woqmc7k1Kqrd0irsBBJcRIdSZt+aVswV8WjZJ9G+SULbloz21+1oRiNq5ATUxKtg+FjUBVPK7x4bht3PhzER/s0+SjnIbOCxyZEs/zqLp5JPcFnfAO4d18vT2ln7Qy4H8it55CcR9Y7rKKWI7e1f772vjw5i56ky/rYjl+Fhfgyw1ZyF89WRYooqXdw45OKLem8faWfrsWL+8n02q6+/xBO4ROuUOlw8ueUEx4uqePaaflxia/45RaLlpFusBXpK87e9qQALatho1NU3ooaOAr0ebff2moCTtA7twD4oyAedDp+gYGJ6B9RaeNmYH9cnyNfAtYOCMOl1bMosYmNGISaDorDSyavbTzN9cBA/Hdb8M2iUUowO92Pz4WK2nyzl6igregVrvsvGajJwV2zoRceWDDpFRKCR9ekFGHSKEb3qtl7q+/6UV7sw6OrfMqk7aM+fucJKJ499cZwjhZWYDTq+PlbS6i2FLqanfC7IbLEmkOByXmfWSSmFCglDjRyPir8JNWgI+PjA8SOQ8iXa15vQkj5CO5CKstlR9rCL3rO++uh1imFhflzV38Kxwio+ySjkq6MlDLCZePgnfVo8I8xk0DHQVjMuVFTlxNdHxwepZ5g7OpSoJi7ijLAYOV5UxaaDRVzR34LlR904F9an1OFi7a5cln15kqIqF+P6tPwMngu53Bq7TpVhMujw9Wn/1lN7/czlllWz5IvjZJdW8/u4SKYNtrHxQCE7T5URd4m11dPLG9JTPhckuDSBBJfzukqdlE6HCotAjRyPbsoM1OTpqIHR4OcPGfvQNq9HO3EY1T8K5d+yAfAAk564Syz0s5ooq3bzwKTerd7DrFeAEYfLzfr0QlJPV6BTivsn9W7WKvxhYX58dqCQPTllBJj02P0Mni4yPz8/SkrL+PxgIc9+eZI9OeUMDDbz/YlS7H4GolpwNPU5mqax9XgJy77MYn16AUmZhdh8DVwSZGrXVtGPv0enShzszimjd4CxwUBf7XLzUdoZnv7yJN8cLaHC6SLEz8ezyPZUiYM/Jh2juMrFY1P6MircH5uvgQE2E+vSCzhSWMkV/SxNXvfUmvp4AwkuLSTB5byuWidlMqMi+qFixqEmTwejqWZx5hfroawUBkSjjHXHSS6+bkfRL8jElAHWOq2Elhrey4+dp8o4VuTg5qHBxDZxbc05vj46egX4sOVwMf93uJiP0s6w93Q5pQ4XJdXw7JbDbMosYlCImUeviuTnI0JIz6vgkwOFxPb2b3KX4YV+yC5j+ddZrEsvwOar564xYeSXO1mfXkDmmUqGh/nV6krSNI39uRX8e18+3x0vYXioH0ZDy1o5F36Pth4r5sktJ0g+UswXmUW4NY3+QSZPcNU0je9OlPJM8km+OVbCsDA/HC6NzYeK+Xh/AbuzyyiucvGX77NxuOHJqf0YEnq+1RhhMWI16fl4fwGlDhdjIvzbPHB2tf9DTrdGfrkTg061aD876OZ7i3Um2f7lvO5UJ62oAC1xLdo3SeAXgLp2JipuOsr//Id5Z9Unp9TB+/vOMGeUvcXHPLvcGvvzKth+spSUk6UcL3IAEOZv4K7YMC7vd35Ff3GVi4c2HsHl1lg1/RKCmtACK3W42JFVxheZhezKLsfuZ+AXI+1MHmBFr1O43BqfZBTw5q5cfPSKe8f2YnCImeTDxSQfKeZ0WTVGvcKtafQONPL4lL6E+jc/sNntdrJP5/LWrlwS084QHWLm5qHBfHagkN055fj56Jg2OIjY3v78c28+e3LK6Wc1cs/YXow+O8HiVImDr44W89WRYo4VObD5Gnhyal/6NbAo9W87atYt3T0mjJuG2BoMMNyfrcMAABb0SURBVC63xtHCKjLyK3C5IdTfQJi/D6H+Pg1uRWS328nNzaXSqVFS5aLE4aLM4aJXgA9h/j6NBjOnW8Pl1jC1MFD/uOzJR4p5Z3cup8ucAPgadFjNeoLMBoJ89dwzphdhARf/nsneYi0kweW87lgn7fhh3O+/UbPnmckX9ZNrasZsQsK6ZX0aklPq4IzLxEB/V70fPofOVPLI50cZFGxmaXy/en9LzSl1sO1EKdtOlrIvpxyXBkFmPbcMC2F6dBDGemaoZRU7+NN3p0jNrQBAp2BUuD9xl1iY2DeAg/mVPPvlSXwNOh6bEtn82Vi+Fh79aA+puRXMuNTGvNgwz3jIgfwKPkw9w7fHS3BrEGjS84uRdq4bFNRgl9nxoioCjfpGA6xb03j+q5N8e7wUk75mt4TegUYiAo30CvDhVImD9LwKDuZXUuWq/6PR36jDatLj1kCjplXl0kBDUVzpxOmumy/QqCMq2FzzJ8SMptWU91iRg+NFVWQV1/wCMSTUlzG9A4iN8GeAzdSs7jtN0/j+RClv/5DLsSIHUcFm4qOslDvcFFY6z/5xUVjpbPIvBBJc/n97dx4cZZ0mcPz79pGzk06nO4SQGEIgIATkMAyYxUUFxyp1BophWA92i5UdZjxgHEuKWM4gLjh4wIJj4aKMJZazUzvlVJFa3JpSOYyO4ooERKMIhBAh5O6k0+l0J939/vaPNwlEUHI0Nh2fT9Vb3el+k/496Tf95HcPkiSX82I5JvX1KdQ7pcbETKXQim4kbcE/4XFmGdsHDAOXe3/Kqjz8x4e13DHBwYqiTDpDOhUNHZTX+jh8zsfZ7g+vnNQ4fpRj40c5NsY7Ey87iEFXiv2nPHQEdeaMTr2ob+p0S4An95+lM6Tz2NxspmSeH7LdFgjxyTkfn9b60BUkxZlItppIijNjNWmUHmvB1xniodlZ/GPepbdWqPN28XlDB7NzUiI2V6UrrFNW1cbX3R/q57xB6tu7CCuwmGCMI4EJrkQmuBIZ70wg3mKiwRek0RekwRekoT1Ie1e4e98iI+lqaCQnJWDRg6TEmUmJN44kq4mati4q3QEq3QG+9nQS6t4JwqRBps1Krj2ea+zx6MoYUHGqpRMAe4KZSRmJ6Ar8IZ1AUMcf0ukMKZLjTKQnWnAkWnAkWLAnmHnvdBvHmwNkp8axdKqLG665/Jp1lyPJZZAkuZw3HGJS7kbUnv9Bvfc2dPohMQmuvQ5t0nS0wumDWp35atGf9+eVQ/X8z7EWJrgSOeUOENQVVpNG4YhEZoyyMTPb9p0TOger0Rdk3T5jdNa/XT8Cf1Dn45p2vmryo3fXkBKtJjq6dHzBcO+Ha64jkdXFI7+1Cev7ZPRPBHEkWi5Zi+uP/rxHwbBOdWsXJg2yU+MuWRNt8Yc4UuujvNbHyWY/VrOJRIuJBKtxG2/R8HWFcftDuP1hPIEQugJnkoW7p7i4Jd8+6JGP/YlJkks/SHI5bzjFpPwdpJytpO2j91AVh6G5wXgiJw/TvfejjZsY3QIOQn/en7CuePr9Guq8XUzLSmZ6VjKFI5Ii0o5/Od7OME+VneXL7ia0fEc8P8qxMTM7hbHp50edKaXoCit8QZ387Exa3c1XvGzfl2j9DYV1RVtnGFucOeLDrK/6bY6F+D5piUkk3HAz7QVTUEpB/TlUxWHUO6Xoz5ag3boAbcG9aHHR/485kswmjcfn5kTltVPizTx5yzV8Vt9BniP+orXeemiaRrxFI95iGvQIJtGX2aRdlduBX30lEiKCNE2DkdloI7NR/3AL6o2dqLdLUZ8eNHbSHHtttIs4bMRbTBGb0CliX9STS3t7O1u2bKGxsbHPNsff9Kc//YnDh42dEH/2s59RXFwMwB/+8AcqKyuxWCyMHTuWFStWYLFEPSxxFdISktD++QHU9cXor72A/swatJtuh8xsCAW7jxCEQ2BPRxuZDSNzwOG87B42Qoi+ov4pXFpaypQpU1i4cCGlpaWUlpaydOnSPueUl5dTVVXFs88+SzAY5Mknn2TatGkkJSUxZ84cVq5cCcDzzz/Pvn37+PGPfxyNUESM0CZNw/TkC0YtZv//XnyCyQS6Tm9nZFwcjMiGZJuRgILB88koPgFt4jS0yTOgYBKaZeDzPYQYjqKeXA4ePEjPIgFz585l3bp1FyWXs2fPMnHiRMxmM2azmdzcXI4cOUJxcTEzZszoPW/cuHE0Nw+fDkJx5fTWYhb9CygdLBYwW41bAE8L1Neg6mqgrgZVXwN+n7FCQJINLBY0ixXV1orauxv19i6IT4SJU9GmXG9sHzCAXTeFGG6inlw8Hg8OhwOAtLQ0PB7PReeMHj2av/71r/zkJz+hs7OTiooKcnL6dlyGQiHef/99li1b9q2vtWfPHvbs2QPA008/jcvlGlSZLRbLoL/3ajXcYup3PN92TkYGjBvfr9fS/T66jh6iq/wjOg8fQD/yEeq//pO4wunEF99Cwuy5mNIuv/T+dxlu7w8Mv5iGWzwwtJi+l+Syfv16WltbL3r8rrvu6vO1pl16+fCpU6dSWVnJb3/7W1JTUxk/fjymb7SB//GPf2TixIlMnPjtw0znz5/P/Pnze78e7LDB4TRst8dwi+l7j2fsJONY/K+Yak6jPvmArkMf0PXSc3hf3gzjC41mtQ6fcfi7b61WcGWiOUeAKxOcI9CcGZCcapxvS4H4RDIyMobV+wNyzcWCq34o8u9+97tvfc5ut9PS0oLD4aClpYXU1EvP1F20aBGLFi0CjL6VrKys3ufeeOMN2traWLFiRWQLLsQAaZoGOWPQcsagFtwLNdXGzptHPoa2VmN1Z7sDLSsHEpOhqxPVVI86+SV8/D6oC/p6epgtNKalowomweTrjQmhtkv/nQhxtYh6s1hRURFlZWUsXLiQsrIyZs6cedE5uq7j8/lISUmhurqar7/+mqlTpwKwd+9ePv30U9auXXtRbUaIaDISTR5aTh4suPey56tQCFqajMPXjvJ5ofuIa28jcPj/4KN3UZpmrAZdOAOSUyDYCV1dEOw+UtPQxhdCXoEMMBBRE/XksnDhQrZs2cK+fft6hyIDVFZW8s477/CrX/2KUCjE2rVrAWMJ6JUrV2LuXjNqx44dZGRk8PjjjwMwa9YsFi9eHJ1ghBgCzWKBjJHGAVzYQGx3uehqqIfqStRnn6A+L0e9+d9w4QIbFgtY48DfYdR+4uIg/1q0gkK0vHFGrSkhyVgWJyEREpKGzdpr4uojy78Mwg+lbTWW/RDiUR0+CIeNJGK1opmMRKG8HjhRgTpegTr+OZw93TcJXSgx2ejXsaWCLdVobktNgzQHpDrQ7Olgd4DDhRb/3SsaKD0M7V5ITulX0vohvEex7qrvcxFCRJ6WlHzpx1PsMKMYbYYx0Vh1tEPtWQj4IdCBCvjB32EcPi9421DtbeBpQdWcNvqGQsY+IH1SUlo6ZIxEG5EFGVlG7aexDtVQC4210FhvTEDVTMa5DieawwXpLkhzGn1NaU7jObvjyv5yRNRJchFimNOSbHDBMjeXW9FLKQUd7dDaAh43ytNiLP7ZnUjU54fBs9c4OT4BRmRB9mi06bPB7gRvK7ibUC1NqDNV8NlBo0+IvsmqPj6he0RcqlHbsaVCugtt7EQYN9FIkkOglIL2NtB1NElm3ztJLkKIPjRNMwYKJKdAdu4lk5HqDBjbGqSkXXbPEKWUMfS61W3Ujlrd4HGTGOrC39hg1Jra21DNjXD4AOqtXcY3jhiFVjAR8ieguTLB4TJqQwlJF5SjE9wN0NSAaqqHpnpUTy2qqc6orQGMykWbPAOtcDoUFKJZ+7f1gOoMgK5DQmLEt0Ue7iS5CCEGTItPMGot/TlX04xVDZJsxod89+MpLhed3+xHCgah+iSq8kvUiS9Qn34MH+zt2zyXmGQ0q/nawfuNSdcWqzEgwpWJNmGyMXcoHDJWxd73JurtUqOPalwhWmYWOEdAegZaeobRdNdUh6quhK8rjdv6GqO/Ki7eeE17OtjTjOa9jJFG0us5ABUOg68NvG3g9Rj9X61uYwRgqxvV0gytzcYSQz3NhHaHcSSngNkMJjOYTN39VhoqFISuTmO5oZ5RgaGg0d8WCkG4+36SDTJHoWVmQ+YoSLGf3+pA1yHQYfzOOtph1Gg065UdSSjJRQhx1dCsVqNJbNxEuG2RUetpbuhtZqO1GVqaUa3NRnNf9+RTYxLqCGMQwqWmJNy2yKiFHP/cGGl3vAJVddyoUcHFc4vSnDB6LNrMOUYS7WkibGs15i59dsiYo3TBtzQkJaP8HZcePBEXZ/zMNCda/rXGfCaP20hgnhajFvgN3znSStPAbDFGCFosRkLytRuJtOecxCSjydHXbsR5QblM//4iZF3Z7RkkuQghrlqapvXWDIbaKKXFJ8CUIrQpRb2PqQ4fuBvB3YhqaTZqMKPz0VK/u49GKWXUmprqe5vjEroCBExmSLGDzY6WatySlg5Jyd/ZrKYCfiMJ6GGjGU7Xz9+3WsEa3zsqEEscmM0X/TwVDhuJuOEcqv6cUevy+SA52ajVJKdAkg0tOdko0xUmyUUI8YOlJSUb839y8gaUvDRNM4Zsp6ah5U8AINXlomuQQ5G1hERj9N0QaGazMbhiRBba5OuH9LMiQaa0CyGEiDhJLkIIISJOkosQQoiIk+QihBAi4iS5CCGEiDhJLkIIISJOkosQQoiIk+QihBAi4n7Q+7kIIYS4MqTmMgglJSXRLkLEDbeYJJ6r33CLabjFA0OLSZKLEEKIiJPkIoQQIuLM69atWxftQsSi/Pz8aBch4oZbTBLP1W+4xTTc4oHBxyQd+kIIISJOmsWEEEJEnCQXIYQQESebhQ3QkSNHePXVV9F1nXnz5rFw4cJoF2lAXnzxRcrLy7Hb7WzevBmA9vZ2tmzZQmNjIxkZGfzmN7/BZrNFuaT909TUxLZt22htbUXTNObPn8/tt98e0zF1dXXxxBNPEAqFCIfDzJ49myVLltDQ0MDWrVvxer3k5+ezcuVKLJbY+RPWdZ2SkhLS09MpKSmJ+XgefPBBEhISMJlMmM1mnn766Zi+7nw+H9u3b+fMmTNomsb999/PqFGjBh+PEv0WDofVQw89pOrq6lQwGFSPPvqoOnPmTLSLNSAVFRWqsrJSPfLII72Pvf7662rXrl1KKaV27dqlXn/99WgVb8DcbreqrKxUSinV0dGhVq1apc6cORPTMem6rvx+v1JKqWAwqB577DH11Vdfqc2bN6u///3vSimlXnrpJfXWW29Fs5gDtnv3brV161a1ceNGpZSK+XgeeOAB5fF4+jwWy9fdCy+8oPbs2aOUMq679vb2IcUjzWIDcPLkSUaOHElmZiYWi4Xi4mIOHjwY7WINyKRJky76z+PgwYPMnTsXgLlz58ZUTA6Ho3c0S2JiItnZ2bjd7piOSdM0EhISAAiHw4TDYTRNo6KigtmzZwNw0003xVRMzc3NlJeXM2/ePMDYgz6W4/k2sXrddXR08OWXX3LLLbcAYLFYSE5OHlI8sVMHvQq43W6cTmfv106nkxMnTkSxRJHh8XhwOBwApKWl4fF4olyiwWloaKCqqopx48bFfEy6rrNmzRrq6uq47bbbyMzMJCkpCbPZDEB6ejputzvKpey/nTt3snTpUvx+PwBerzem4+nx1FNPAXDrrbcyf/78mL3uGhoaSE1N5cUXX6S6upr8/HyWLVs2pHgkuYg+NE1D07RoF2PAAoEAmzdvZtmyZSQlJfV5LhZjMplMPPfcc/h8PjZt2sS5c+eiXaRBO3ToEHa7nfz8fCoqKqJdnIhZv3496enpeDweNmzYwKhRo/o8H0vXXTgcpqqqivvuu4+CggJeffVVSktL+5wz0HgkuQxAeno6zc3NvV83NzeTnp4exRJFht1up6WlBYfDQUtLC6mpqdEu0oCEQiE2b97MjTfeyKxZs4DYj6lHcnIyhYWFHD9+nI6ODsLhMGazGbfbHTPX3ldffcUnn3zC4cOH6erqwu/3s3PnzpiNp0dPee12OzNnzuTkyZMxe905nU6cTicFBQUAzJ49m9LS0iHFI30uAzB27Fhqa2tpaGggFArx4YcfUlRUFO1iDVlRURFlZWUAlJWVMXPmzCiXqP+UUmzfvp3s7GzuvPPO3sdjOaa2tjZ8Ph9gjBw7evQo2dnZFBYW8tFHHwHw7rvvxsy1d88997B9+3a2bdvGww8/zOTJk1m1alXMxgNGTbmniS8QCHD06FFyc3Nj9rpLS0vD6XT21pA/++wzcnJyhhSPzNAfoPLycl577TV0Xefmm29m0aJF0S7SgGzdupUvvvgCr9eL3W5nyZIlzJw5ky1bttDU1BRzwyePHTvG2rVryc3N7a2y33333RQUFMRsTNXV1Wzbtg1d11FKccMNN7B48WLq6+vZunUr7e3tjBkzhpUrV2K1WqNd3AGpqKhg9+7dlJSUxHQ89fX1bNq0CTCalObMmcOiRYvwer0xe92dPn2a7du3EwqFGDFiBA888ABKqUHHI8lFCCFExEmzmBBCiIiT5CKEECLiJLkIIYSIOEkuQgghIk6SixBCiIiT5CJEDFiyZAl1dXXRLoYQ/SYz9IUYoAcffJDW1lZMpvP/m910000sX748iqW6tLfeeovm5mbuuecennjiCe677z5Gjx4d7WKJHwBJLkIMwpo1a7juuuuiXYzLOnXqFDNmzEDXdWpqasjJyYl2kcQPhCQXISLo3XffZe/eveTl5fHee+/hcDhYvnw5U6ZMAYyVtXfs2MGxY8ew2WwsWLCA+fPnA8ZKyKWlpezfvx+Px0NWVharV6/G5XIBcPToUX7/+9/T1tbGnDlzWL58+WUXEjx16hSLFy/m3LlzZGRk9K5CLMSVJslFiAg7ceIEs2bN4pVXXuHjjz9m06ZNbNu2DZvNxvPPP88111zDSy+9xLlz51i/fj0jR45k8uTJvPnmm3zwwQc89thjZGVlUV1dTXx8fO/PLS8vZ+PGjfj9ftasWUNRURHTpk276PWDwSC/+MUvUEoRCARYvXo1oVAIXddZtmwZP/3pT2Nu2SIReyS5CDEIzz33XJ9awNKlS3trIHa7nTvuuANN0yguLmb37t2Ul5czadIkjh07RklJCXFxceTl5TFv3jzKysqYPHkye/fuZenSpb1Lt+fl5fV5zYULF5KcnNy7UvLp06cvmVysVis7d+5k7969nDlzhmXLlrFhwwbuuusuxo0bd+V+KUJcQJKLEIOwevXqb+1zSU9P79NclZGRgdvtpqWlBZvNRmJiYu9zLpeLyspKwNjCITMz81tfMy0trfd+fHw8gUDgkudt3bqVI0eO0NnZidVqZf/+/QQCAU6ePElWVhYbN24cUKxCDIYkFyEizO12o5TqTTBNTU0UFRXhcDhob2/H7/f3JpimpqbefUGcTif19fXk5uYO6fUffvhhdF1nxYoVvPzyyxw6dIgDBw6watWqoQUmxADIPBchIszj8fC3v/2NUCjEgQMHqKmpYfr06bhcLiZMmMCf//xnurq6qK6uZv/+/dx4440AzJs3j7/85S/U1tailKK6uhqv1zuoMtTU1JCZmYnJZKKqqoqxY8dGMkQhLktqLkIMwjPPPNNnnst1113H6tWrASgoKKC2tpbly5eTlpbGI488QkpKCgC//vWv2bFjB7/85S+x2Wz8/Oc/721eu/POOwkGg2zYsAGv10t2djaPPvrooMp36tQpxowZ03t/wYIFQwlXiAGT/VyEiKCeocjr16+PdlGEiCppFhNCCBFxklyEEEJEnDSLCSGEiDipuQghhIg4SS5CCCEiTpKLEEKIiJPkIoQQIuIkuQghhIi4/wc+HcWypZNkcwAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "# Step 7: Print Output\n",
    "# plot the training + testing loss and accuracy\n",
    "Fepochs = len(H.history['loss'])\n",
    "plt.style.use(\"ggplot\")\n",
    "plt.figure()\n",
    "plt.plot(np.arange(0, Fepochs), H.history[\"loss\"], label=\"train_loss\")\n",
    "plt.plot(np.arange(0, Fepochs), H.history[\"val_loss\"], label=\"val_loss\")\n",
    "plt.title(\"Training Loss\")\n",
    "plt.xlabel(\"Epoch #\")\n",
    "plt.ylabel(\"Loss\")\n",
    "plt.legend()\n",
    "figpath_final = os.path.join(config['output_dir'], config['input_type'] + \"_loss.png\")\n",
    "plt.savefig(figpath_final)\n",
    "if config['show_plots']:\n",
    "    plt.show()\n",
    "\n",
    "plt.figure()\n",
    "plt.plot(np.arange(0, Fepochs), H.history[\"dice_coefficient_monitor\"], label=\"train_DSC\")\n",
    "plt.plot(np.arange(0, Fepochs), H.history[\"val_dice_coefficient_monitor\"], label=\"val_DSC\")\n",
    "plt.title(\"Training DSC\")\n",
    "plt.xlabel(\"Epoch #\")\n",
    "plt.ylabel(\"DSC\")\n",
    "plt.legend()\n",
    "figpath_final = os.path.join(config['output_dir'], config[\"input_type\"] + '_DSC.png')\n",
    "plt.savefig(figpath_final)\n",
    "if config['show_plots']:\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "file_extension": ".py",
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.9"
  },
  "mimetype": "text/x-python",
  "name": "python",
  "npconvert_exporter": "python",
  "pygments_lexer": "ipython3",
  "version": 3
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
