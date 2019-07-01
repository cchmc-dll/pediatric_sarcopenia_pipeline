import numpy as np
import keras
from keras.utils import np_utils
from pyimagesearch.utils.generator_utils_Elan import convert_data, add_data_classify,add_data_segment_2D,add_data_segment
from sklearn import preprocessing
from keras.preprocessing.image import ImageDataGenerator

class DataGenerator_3D_Segmentation(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data_file, index_list, n_labels=1, labels = None,  
                 batch_size=16, patch_shape= None,patch_overlap=0, patch_start_offset=None,
                 augment=False, augment_flip=True, augment_distortion_factor=0.25,
                 shuffle_index_list=True, skip_blank=True, permute=False, reduce = 0):
        'Initialization'
        self.patch_shape = patch_shape
        self.batch_size = batch_size
        self.labels = labels
        self.index_list = index_list
        self.n_labels = n_labels
        self.shuffle = shuffle_index_list
        self.skip_blank = skip_blank
        self.data_file = data_file
        self.patch_shape = patch_shape
        self.patch_overlap = patch_overlap
        self.patch_offset = patch_start_offset
        self.permute = permute
        self.augment = augment
        self.augment_flip = augment_flip
        self.augment_distortion_factor = augment_distortion_factor
        self.reduce = reduce
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.index_list) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        if((index+1)*self.batch_size <= len(self.index_list)):
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        else:
            indexes = self.indexes[index*self.batch_size:len(self.index_list)]

        # Find list of IDs
        index_list_temp = [self.index_list[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(index_list_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.index_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, index_list_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        x_list = list()
        y_list = list()
        # Generate data
        for i,  indexx in enumerate(index_list_temp):
            add_data_segment(x_list, y_list, self.data_file, indexx, augment=self.augment, augment_flip=self.augment_flip,
                        augment_distortion_factor=self.augment_distortion_factor, patch_shape=self.patch_shape,
                        skip_blank=self.skip_blank, permute=self.permute, reduce = self.reduce)
            
        return convert_data(x_list, y_list, n_labels=self.n_labels, labels=self.labels)

class DataGenerator_2D_Segmentation(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data_file, index_list,batch_size = 16, n_labels=1, labels = None,shuffle_index_list = True):
        'Initialization'
        self.data_file = data_file
        self.batch_size = batch_size
        self.labels = labels
        self.index_list = index_list
        self.n_labels = n_labels
        self.shuffle = shuffle_index_list
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.index_list) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        if((index+1)*self.batch_size <= len(self.index_list)):
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        else:
            indexes = self.indexes[index*self.batch_size:len(self.index_list)]

        # Find list of IDs
        index_list_temp = [self.index_list[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(index_list_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.index_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, index_list_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        x_list = list()
        y_list = list()
        # Generate data
        for i,  indexx in enumerate(index_list_temp):
            add_data_segment_2D(x_list, y_list, self.data_file, indexx)

            
        return convert_data(x_list, y_list, n_labels=self.n_labels, labels=self.labels)

class DataGenerator_3DCL_Segmentation(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,problem_type, data_file, index_list, n_labels=1, labels = None,  
                 batch_size=16, patch_shape= None,patch_overlap=0, patch_start_offset=None,
                 augment=False, augment_flip=True, augment_distortion_factor=0.25,
                 shuffle_index_list=True, skip_blank=True, permute=False, reduce = 0):
        'Initialization'
        self.problem_type = problem_type
        self.patch_shape = patch_shape
        self.batch_size = batch_size
        self.labels = labels
        self.index_list = index_list
        self.n_labels = n_labels
        self.shuffle = shuffle_index_list
        self.skip_blank = skip_blank
        self.data_file = data_file
        self.patch_shape = patch_shape
        self.patch_overlap = patch_overlap
        self.patch_offset = patch_start_offset
        self.permute = permute
        self.augment = augment
        self.augment_flip = augment_flip
        self.augment_distortion_factor = augment_distortion_factor
        self.reduce = reduce
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.index_list) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        if((index+1)*self.batch_size <= len(self.index_list)):
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        else:
            indexes = self.indexes[index*self.batch_size:len(self.index_list)]

        # Find list of IDs
        index_list_temp = [self.index_list[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(index_list_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.index_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, index_list_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        x_list = list()
        y_list = list()
        # Generate data
        for i,  indexx in enumerate(index_list_temp):
            add_data_reduce(x_list, y_list, self.data_file, indexx, augment=self.augment, augment_flip=self.augment_flip,
                            augment_distortion_factor=self.augment_distortion_factor, patch_shape=self.patch_shape,
                            skip_blank=self.skip_blank, permute=self.permute, reduce = self.reduce)
            
        return convert_data(x_list, y_list, n_labels=self.n_labels, labels=self.labels)

class DataGenerator_CL_Classification(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,data_file, index_list, n_classes=2,classes = ['abnormal','normal'],
                 batch_size=16, shuffle_index_list=True):
        'Initialization'
        self.data_file = data_file
        self.classes = classes
        self.n_classes = n_classes
        le = preprocessing.LabelEncoder()
        self.encoder = le.fit(np.asarray(classes))
        self.batch_size = batch_size
       
        self.index_list = index_list
       
        self.shuffle = shuffle_index_list
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.index_list) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        if((index+1)*self.batch_size <= len(self.index_list)):
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        else:
            indexes = self.indexes[index*self.batch_size:len(self.index_list)]

        # Find list of IDs
        index_list_temp = [self.index_list[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(index_list_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.index_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __data_generation(self, index_list_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        x_cl_list = list()
        y_list = list()
        # Generate data
        for i,  indexx in enumerate(index_list_temp):
            x_cl_list.append(self.data_file.root.cldata[indexx,:])
            y_list.append(self.data_file.root.truth[indexx])
  
        x_cl = np.asarray(x_cl_list)
        y = np.asarray([y.decode("utf-8") for y in y_list])
        y  = np_utils.to_categorical(self.encoder.transform(y), self.n_classes)
        return x_cl, y

class DataGenerator_3DCL_Classification(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,data_file,index_list,batch_size=16,n_classes=2,classes = ['abnormal','normal'],
                 augment=False, augment_flip=True, augment_distortion_factor=0.25,
                 shuffle_index_list=True, skip_blank=True, permute=False, reduce = 0):
        'Initialization'
        self.batch_size = batch_size
        self.classes = classes
        self.n_classes = n_classes
        le = preprocessing.LabelEncoder()
        self.encoder = le.fit(np.asarray(classes))
        self.index_list = index_list
        self.shuffle = shuffle_index_list
        self.skip_blank = skip_blank
        self.data_file = data_file
        self.permute = permute
        self.augment = augment
        self.augment_flip = augment_flip
        self.augment_distortion_factor = augment_distortion_factor
        self.reduce = reduce
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.index_list) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        if((index+1)*self.batch_size <= len(self.index_list)):
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        else:
            indexes = self.indexes[index*self.batch_size:len(self.index_list)]

        # Find list of IDs
        index_list_temp = [self.index_list[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(index_list_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.index_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, index_list_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        x_list = list()
        x_cl_list = list()
        y_list = list()
        # Generate data
        for i,  indexx in enumerate(index_list_temp):
            add_data_classify(x_list, self.data_file, indexx, augment=self.augment, augment_flip=self.augment_flip,
                            augment_distortion_factor=self.augment_distortion_factor,
                            skip_blank=self.skip_blank, permute=self.permute, reduce = self.reduce)
            x_cl_list.append(self.data_file.root.cldata[indexx,:])
            y_list.append(self.data_file.root.truth[indexx])
   
        x_im = np.asarray(x_list)
        x_cl = np.asarray(x_cl_list)
        y = np.asarray([y.decode("utf-8") for y in y_list])
        y  = np_utils.to_categorical(self.encoder.transform(y), self.n_classes)
        x = [x_cl,x_im]
        return x, y

class DataGenerator_3D_Classification(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,data_file,index_list,batch_size=16,n_classes=2,classes = ['abnormal','normal'],
                 augment=False, augment_flip=True, augment_distortion_factor=0.25,
                 shuffle_index_list=True, skip_blank=True, permute=False, reduce = 0):
        'Initialization'
        self.batch_size = batch_size
        self.classes = classes
        self.n_classes = n_classes
        le = preprocessing.LabelEncoder()
        self.encoder = le.fit(np.asarray(classes))
        self.index_list = index_list
        self.shuffle = shuffle_index_list
        self.skip_blank = skip_blank
        self.data_file = data_file
        self.permute = permute
        self.augment = augment
        self.augment_flip = augment_flip
        self.augment_distortion_factor = augment_distortion_factor
        self.reduce = reduce
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.index_list) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        if((index+1)*self.batch_size <= len(self.index_list)):
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        else:
            indexes = self.indexes[index*self.batch_size:len(self.index_list)]

        # Find list of IDs
        index_list_temp = [self.index_list[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(index_list_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.index_list))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, index_list_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        x_list = list()
        y_list = list()
        # Generate data
        for i,  indexx in enumerate(index_list_temp):
            add_data_classify(x_list, self.data_file, indexx, augment=self.augment, augment_flip=self.augment_flip,
                            augment_distortion_factor=self.augment_distortion_factor,
                            skip_blank=self.skip_blank, permute=self.permute, reduce = self.reduce)
            y_list.append(self.data_file.root.truth[indexx])
   
        x_im = np.asarray(x_list)
        y = np.asarray([y.decode("utf-8") for y in y_list])
        y  = np_utils.to_categorical(self.encoder.transform(y), self.n_classes)
        return x_im, y

class TestGenerator_3D_Classification(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,data_file,index_list,batch_size=16,reduce = 0):
        'Initialization'
        self.batch_size = batch_size
        self.index_list = index_list
        self.data_file = data_file
        self.reduce = reduce
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.index_list) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        if((index+1)*self.batch_size <= len(self.index_list)):
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        else:
            indexes = self.indexes[index*self.batch_size:len(self.index_list)]

        # Find list of IDs
        index_list_temp = [self.index_list[k] for k in indexes]

        # Generate data
        X = self.__data_generation(index_list_temp)

        return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.index_list))

    def __data_generation(self, index_list_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        x_list = list()
        # Generate data
        for i,  indexx in enumerate(index_list_temp):
            add_data_classify(x_list, self.data_file, indexx, reduce = self.reduce)
           
        x_im = np.asarray(x_list)
        return x_im

class TestGenerator_3DCL_Classification(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,data_file,index_list,batch_size=16,reduce = 0):
        'Initialization'
        self.batch_size = batch_size
        self.index_list = index_list
        self.data_file = data_file
        self.reduce = reduce
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.index_list) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        if((index+1)*self.batch_size <= len(self.index_list)):
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        else:
            indexes = self.indexes[index*self.batch_size:len(self.index_list)]

        # Find list of IDs
        index_list_temp = [self.index_list[k] for k in indexes]

        # Generate data
        X= self.__data_generation(index_list_temp)
        return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.index_list))

    def __data_generation(self, index_list_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        x_list = list()
        x_cl_list = list()
        # Generate data
        for i,  indexx in enumerate(index_list_temp):
            add_data_classify(x_list, self.data_file, indexx, reduce = self.reduce)
            x_cl_list.append(self.data_file.root.cldata[indexx,:])
   
        x_im = np.asarray(x_list)
        x_cl = np.asarray(x_cl_list)
        x = [x_cl,x_im]
        return x

class TestGenerator_CL_Classification(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,data_file,index_list,batch_size=16):
        'Initialization'
        self.batch_size = batch_size
        self.index_list = index_list
        self.data_file = data_file
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.index_list) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        if((index+1)*self.batch_size <= len(self.index_list)):
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        else:
            indexes = self.indexes[index*self.batch_size:len(self.index_list)]

        # Find list of IDs
        index_list_temp = [self.index_list[k] for k in indexes]

        # Generate data
        X= self.__data_generation(index_list_temp)
        return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.index_list))

    def __data_generation(self, index_list_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        x_cl_list = list()
        # Generate data
        for i,  indexx in enumerate(index_list_temp):
            x_cl_list.append(self.data_file.root.cldata[indexx,:])
   
        x_cl = np.asarray(x_cl_list)
        return x_cl

