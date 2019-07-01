import tables
import os
import glob
import cmd
import numpy as np
from unet3d.utils.utils import pickle_dump, pickle_load

config = dict()
config["h5_file"] = "CT_190PTS.h5"
config["pickle_file"] = "validation_0.2.pkl"



h5_file = os.path.abspath(os.path.join("datasets", config["h5_file"]))
pkl_file = os.path.abspath(os.path.join('datasets',config['pickle_file']))

data = tables.open_file(h5_file,mode='r')
pkl_list = pickle_load(pkl_file)


truth = data.root.truth
subject_ids = data.root.subject_ids

print('Number of Samples: ', truth.shape[0])

for index in pkl_list:
    print(index,' Subject_ids is: ',subject_ids[index])
    arr = truth[index][0]
    print('arr type:', arr.dtype, 'arr max: ', np.max(arr), 'arr min: ', np.min(arr))





