import os
import glob
import cmd
import shutil




def create_dir(path):
     try:  
        os.mkdir(path)
     except OSError:  
        print ("Creation of the directory %s failed" % path)
     else:  
        print ("Successfully created the directory %s " % path)


# Get INPUT/ OUTPUT
config = dict()
config['input_img'] = os.path.abspath("Validation_Input")
config['input_label'] =  os.path.abspath("Validation_L3_manual")
config['output'] = os.path.abspath("ImageData")



# Parse Folder and get patient ID, raw image and label
 
img_files = list()
label_files = list()
subject_ids = list()
for subject_dir in glob.glob(os.path.join(config['input_img'], "*")):
    subject_ids.append(os.path.basename(subject_dir))
    img_files.append(os.path.join(subject_dir, "preprocessed.tif"))
    label_files.append(os.path.join(config['input_label'],os.path.basename(subject_dir), "PP_V3.tif"))
    
for i,subject in enumerate(subject_ids):
    path = os.path.join(config['output'],subject)
    create_dir(path)
    shutil.move(img_files[i], os.path.join(path,"CT.tif"))
    shutil.move(label_files[i], os.path.join(path,"Muscle.tif.bak.bak"))


