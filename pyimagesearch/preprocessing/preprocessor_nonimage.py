from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class preprocess_nonimaging:
    """description of class: Preprocesses the non imaging input features. For training data, normalization occurs over all the samples and the models used are saved to normalize testing data""" 
    def __init__(self, train_flag=1, filenames={'scaler':'minmax_scaler.save'}):
        train_flag = self.train_flag
        filenames = self.filenames

    def preprocess(self,input_data):
        if train_flag:
            cs = MinMaxScaler()
            input_scaled = cs.fit_transform(input_data)
            joblib.dump(cs,filenames['scaler'])
            return (input_scaled)
        else:
            cs = joblib.load(filenames['scaler'])
            input_scaled = cs.transform(input_data)
            return (input_scaled)