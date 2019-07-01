# import the necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers import LeakyReLU

class MLP:
    @staticmethod
    def build(dim,num_outputs,branch=False):
	# define our MLP network
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(dim,)))
        model.add(Dropout(0.4))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(num_outputs))
        if branch:
            return model
        else:
            model.add(Activation("softmax"))
            return model


class MLP10:
    @staticmethod
    def build(dim,num_outputs,branch=False):
	# define our MLP network
        model = Sequential()
        model.add(Dense(14, activation='relu', input_shape=(dim,)))
        model.add(Dropout(0.2))
        model.add(Dense(6))
        model.add(Dropout(0.2))
        model.add(LeakyReLU(alpha=0.03))
        model.add(Dense(num_outputs))
        if branch:
            return model
        else:
            model.add(Activation("softmax"))
            return model
