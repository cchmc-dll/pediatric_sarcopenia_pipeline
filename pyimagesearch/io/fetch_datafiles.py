import os
import glob
import pandas as pd

class fetch_datafiles():
	def __init__(self,problem_type='Classification',input_type='both',input_images=None,input_clinical=None):
        self.problem_type = problem_type
        self.input_type = input_type
        self.input_images = input_images
        self.input_clinical = input_clinical
        
    def get_files():
        if problem_type is 'Segmentation':
            training_data_files = list()
            subject_ids = list()
            for subject_dir in glob.glob(input_images, "*"):
                subject_ids.append(os.path.basename(subject_dir))
                subject_files = list()
                for modality in config["training_modalities"] + ["Label"]:
                    subject_files.append(os.path.join(subject_dir, modality + ".nii"))
                training_data_files.append(tuple(subject_files))

            if input_type is 'both':
                features =  pd.read_csv(os.path.join(input_clinical,'Features.csv'))
                truth    = pd.read_csv(os.path.join(input_clinical,'Truth.csv'))


             return training_data_files, subject_ids
            
        if problem_type is 'Classification':


        if problem_type is 'Regression':