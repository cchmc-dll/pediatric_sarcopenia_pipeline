# import the necessary packages
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers.convolutional import Conv2DTranspose
from keras.layers import Input
from keras.layers import concatenate
from keras.layers import UpSampling2D
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras import backend as K

# Model based on: https://github.com/zhixuhao/unet/blob/master/model.py
class Unet2D:
	@staticmethod
	def build(inputShape, classes):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		# inputShape = (height, width, depth)
		# if we are using "channels first", update the input shape
		# and channels dimension
		chanDim = -1
		if K.image_data_format() == "channels_first":
			#inputShape = (depth, height, width)
			chanDim = 1
		inputs = Input(inputShape)
        #Layer 1
		conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
		conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        #Layer 2
		conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
		conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        #Layer 3
		conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
		conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
		pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        #Layer 4
		conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
		conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
		drop4 = Dropout(0.5)(conv4)
		pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
		#Layer 5
		conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
		conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
		drop5 = Dropout(0.5)(conv5)

        #Layer 6
		up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
		merge6 = concatenate([drop4,up6], axis = chanDim)
		conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
		conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
        #Layer 7
		up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
		merge7 = concatenate([conv3,up7], axis = chanDim)
		conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
		conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
        #Layer 8
		up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
		merge8 = concatenate([conv2,up8], axis = chanDim)
		conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
		conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
        #Layer 9
		up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
		merge9 = concatenate([conv1,up9], axis = chanDim)
		conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
		conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
		conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        #Layer 10
		conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

		model = Model(input = inputs, output = conv10)

		# return the constructed network architecture
		return model


#Unet with BatchNorm based on: https://www.depends-on-the-definition.com/unet-keras-segmenting-images/
class Unet2D_BN:
    @staticmethod
    def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
        # first layer
        x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
                   padding="same")(input_tensor)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation("relu")(x)
        # second layer
        x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
                   padding="same")(x)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation("relu")(x)
        return x

    @staticmethod
    def build(inputShape, classes):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		# inputShape = (height, width, depth)
		# if we are using "channels first", update the input shape
		# and channels dimension

        chanDim = -1
        if K.image_data_format() == "channels_first":
			#inputShape = (depth, height, width)
            chanDim = 1

        input_img = Input(inputShape)
        n_filters=16
        dropout=0.5
        batchnorm=True
        # contracting path
        c1 = Unet2D_BN.conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
        p1 = MaxPooling2D((2, 2)) (c1)
        p1 = Dropout(dropout*0.5)(p1)

        c2 = Unet2D_BN.conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
        p2 = MaxPooling2D((2, 2)) (c2)
        p2 = Dropout(dropout)(p2)

        c3 = Unet2D_BN.conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
        p3 = MaxPooling2D((2, 2)) (c3)
        p3 = Dropout(dropout)(p3)

        c4 = Unet2D_BN.conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
        p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
        p4 = Dropout(dropout)(p4)
    
        c5 = Unet2D_BN.conv2d_block(p4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)
    
        # expansive path
        u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (c5)
        u6 = concatenate([u6, c4],axis = chanDim)
        u6 = Dropout(dropout)(u6)
        c6 = Unet2D_BN.conv2d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

        u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c6)
        u7 = concatenate([u7, c3],axis = chanDim)
        u7 = Dropout(dropout)(u7)
        c7 = Unet2D_BN.conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

        u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c7)
        u8 = concatenate([u8, c2],axis = chanDim)
        u8 = Dropout(dropout)(u8)
        c8 = Unet2D_BN.conv2d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

        u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)
        u9 = concatenate([u9, c1],axis = chanDim)
        u9 = Dropout(dropout)(u9)
        c9 = Unet2D_BN.conv2d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    
        outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)
        model = Model(inputs=[input_img], outputs=[outputs])
        return model


#Unet with BatchNorm based on: https://www.depends-on-the-definition.com/unet-keras-segmenting-images/
# MODIFIED - elan
class Unet2D_BN_MOD:
    @staticmethod
    def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
        # first layer
        x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
                   padding="same")(input_tensor)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation("relu")(x)
        # second layer
        x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
                   padding="same")(x)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation("relu")(x)
        return x

    @staticmethod
    def build(inputShape, classes):
		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		# inputShape = (height, width, depth)
		# if we are using "channels first", update the input shape
		# and channels dimension

        chanDim = -1
        if K.image_data_format() == "channels_first":
			#inputShape = (depth, height, width)
            chanDim = 1

        input_img = Input(inputShape)
        n_filters=16
        dropout=0.5
        batchnorm=True
        # contracting path
        c1 = Unet2D_BN.conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
        p1 = MaxPooling2D((2, 2)) (c1)
        p1 = Dropout(dropout*0.5)(p1)

        c2 = Unet2D_BN.conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
        p2 = MaxPooling2D((2, 2)) (c2)
        p2 = Dropout(dropout)(p2)

        c3 = Unet2D_BN.conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
        p3 = MaxPooling2D((2, 2)) (c3)
        p3 = Dropout(dropout)(p3)

        c4 = Unet2D_BN.conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
        p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
        p4 = Dropout(dropout)(p4)

        c5 = Unet2D_BN.conv2d_block(p4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)
        p5 = MaxPooling2D(pool_size=(2, 2)) (c5)
        p5 = Dropout(dropout)(p5)

        # Additional Layer
        c6 = Unet2D_BN.conv2d_block(p5, n_filters=n_filters*32, kernel_size=3, batchnorm=batchnorm)

        # expansive path
        u7 = Conv2DTranspose(n_filters*16, (3, 3), strides=(2, 2), padding='same') (c6)
        u7 = concatenate([u7, c5],axis = chanDim)
        u7 = Dropout(dropout)(u7)
        c7 = Unet2D_BN.conv2d_block(u7, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)

        u8 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (c7)
        u8 = concatenate([u8, c4],axis = chanDim)
        u8 = Dropout(dropout)(u8)
        c8 = Unet2D_BN.conv2d_block(u8, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

        u9 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c8)
        u9 = concatenate([u9, c3],axis = chanDim)
        u9 = Dropout(dropout)(u9)
        c9 = Unet2D_BN.conv2d_block(u9, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

        u10 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c9)
        u10 = concatenate([u10, c2],axis = chanDim)
        u10 = Dropout(dropout)(u10)
        c10 = Unet2D_BN.conv2d_block(u10, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

        u11 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c10)
        u11 = concatenate([u11, c1],axis = chanDim)
        u11 = Dropout(dropout)(u11)
        c11 = Unet2D_BN.conv2d_block(u11, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)

        outputs = Conv2D(1, (1, 1), activation='sigmoid') (c11)
        model = Model(inputs=[input_img], outputs=[outputs])
        return model

