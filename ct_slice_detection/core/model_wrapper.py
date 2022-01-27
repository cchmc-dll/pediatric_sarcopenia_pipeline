import os
import tensorflow.keras.callbacks as cbks
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.models import load_model
import pandas as pd
from tensorflow.keras import backend as K
from tensorflow.keras.utils import multi_gpu_model

class BaseModelWrapper():

    def __init__(self, config, data_loader, name, is_multi_gpu=False):

        K.clear_session()
        self.config = config
        self.model = None
        self.name = name
        self.start_epoch = 0
        self.is_multi_gpu = is_multi_gpu
        self.callbacks = self.get_default_callbacks()
        self.data_loader = data_loader
        self.custom_objects = {}


    def save(self):
        if self.model is None:
            raise Exception("Model does not exist.")

        print("Saving model...")
        model_path = os.path.join(self.config.model_path, self.name + '.h5')
        self.model.save(model_path)
        print("Model saved")


    def load(self):
        if self.model is None:
            raise Exception("Model not defined.")
        model_path = os.path.join(self.config.model_path, self.name  + '.h5')
        checkpoint_path = os.path.join(self.config.model_path, self.name +"_model-checkpoint.hdf5")
        model_list = []
        if os.path.exists(model_path):
            model_list.append((os.path.getmtime(model_path), model_path))

        if os.path.exists(checkpoint_path):
            model_list.append((os.path.getmtime(checkpoint_path), checkpoint_path))

        model_list.sort(reverse=True)
        print(model_list)
        if model_list != []:
            model_path = model_list[0][1]
            print("Loading model  {} ...\n".format(model_path))
            # self.model = load_model(model_path, custom_objects=self.custom_objects)
            self.model.load_weights(model_path)
            print("Model loaded")
            print(self.model.summary())
        else:
            print("No saved model found.")


    def build_model(self):
        raise NotImplementedError


    def get_default_callbacks(self):

        callbacks_list = []

        # checkpoint callback
        filepath = os.path.join(self.config.model_path, self.name +"_model-checkpoint.hdf5")
        checkpoint = ModelCheckpoint(filepath, verbose=1)

        log_path = os.path.join(self.config.model_path, self.name +"_log.csv")
        csvlogger = CSVLogger(log_path, separator=',', append=True)

        if os.path.exists(log_path):
            try:
                df = pd.read_csv(log_path)
                self.start_epoch = df.epoch.max()
            except:
                self.start_epoch = 0

        callbacks_list.append(csvlogger)

        if self.config.do_checkpoint:
            callbacks_list.append(checkpoint)

        return callbacks_list


    def train(self):

        self.load()

        config = self.config
        self.model.fit(self.data_loader.x_train, self.data_loader.y_train,
                  batch_size=config.batch_size,
                  epochs=config.num_epochs,
                  validation_data=(self.data_loader.x_val, self.data_loader.y_val),
                  callbacks=self.callbacks,
                  shuffle=True)


    def train_generator(self):

        self.load()

        config = self.config

        try:

            history = self.model.fit_generator(self.data_loader.train_generator,
                                          steps_per_epoch=len(self.data_loader.y_train)/config.batch_size,
                                          epochs=config.num_epochs,
                                          # validation_data=self.data_loader.val_generator,
                                          # validation_steps=len(self.data_loader.y_val)/config.batch_size,
                                            callbacks=self.callbacks,
                                            initial_epoch=self.start_epoch
                                            # max_queue_size=20,
                                            # workers=4, use_multiprocessing=True
                                          )

            # self.data_loader.clean_up()
        except KeyboardInterrupt:
            pass

        # finally:
        #     self.data_loader.clean_up()




    def evaluate(self):
        return self.model.evaluate(self.data_loader.x_test,
                                   self.data_loader.y_test,
                                   steps=len(self.data_loader.y_test))


    def predict(self):
        self.model.predict(self.data_loader.x_test, steps=len(self.data_loader.y_test))
