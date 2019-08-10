from keras.applications.vgg16 import VGG16
from keras.layers import *
from keras.models import *
from keras.optimizers import Adam

from ct_slice_detection.core.model_wrapper import BaseModelWrapper





from keras.layers import Layer
from keras.engine import InputSpec


class _GlobalHorizontalPooling2D(Layer):
    """Abstract class for different global pooling 2D layers.
    """

    @interfaces.legacy_global_pooling_support
    def __init__(self, data_format=None, **kwargs):
        super(_GlobalHorizontalPooling2D, self).__init__(**kwargs)
        self.data_format = data_format
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        # if self.data_format == 'channels_last':
        #     return (input_shape[0], input_shape[1], input_shape[2])
        # else:
        return (input_shape[0], input_shape[1], input_shape[3])

    def call(self, inputs):
        raise NotImplementedError

    def get_config(self):
        config = {'data_format': self.data_format}
        base_config = super(_GlobalHorizontalPooling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GlobalMaxHorizontalPooling2D(_GlobalHorizontalPooling2D):
    """Global max pooling operation for spatial data.
    # Arguments
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
    # Input shape
        - If `data_format='channels_last'`:
            4D tensor with shape:
            `(batch_size, rows, cols, channels)`
        - If `data_format='channels_first'`:
            4D tensor with shape:
            `(batch_size, channels, rows, cols)`
    # Output shape
        2D tensor with shape:
        `(batch_size, channels)`
    """

    def call(self, inputs):
        # if self.data_format == 'channels_last':
        return K.max(inputs, axis=[2])
        # else:
        #     return K.max(inputs, axis=[3])


def up_conv_block_add_1D(inp, inp2, num_filters=64, kernel_size=3, momentum=0.8, padding="same", up_size=2,
                               num_blocks=3, is_residual=False):

    def up_conv_unit(inp, num_filters=64, kernel_size=3, momentum=0.8, padding="same", do_act=True):
        conv = Conv1D(num_filters, kernel_size, padding=padding)(inp)
        conv = BatchNormalization(momentum=momentum)(conv)
        if do_act:
            conv = LeakyReLU(0.05)(conv)
        return conv



    if is_residual:
        inp = Conv1D(num_filters, 1, padding=padding)(inp)


    inp = UpSampling1D(size=up_size)(inp)
    upcov = inp
    # inp2 = conv_block(inp2, num_filters=num_filters, kernel_size=1, num_blocks=1, pool_size=None)
    inp2 =  GlobalMaxHorizontalPooling2D()(inp2)
    #     att = Conv2D(1, kernel_size=1, activation='sigmoid')(inp2)
    #     inp2 = merge([inp2, concatenate([att]*num_filters)], mode='mul')
    upcov = concatenate([ upcov, inp2], axis=2)
    #     upcov = Add()([ upcov, inp2])
    upcov = Dropout(0.25)(upcov)
    for i in range(num_blocks):
        upcov = up_conv_unit(upcov, num_filters, kernel_size, momentum, padding, True)
    # if num_blocks == 1:
    upcov = up_conv_unit(upcov, num_filters, 1, momentum, padding, False)

    if is_residual:
        upcov = add([upcov, inp])
    upcov = LeakyReLU(0.05)(upcov)

    return upcov



def conv_block(inp, num_filters=64, kernel_size=3, momentum=0.8, padding="same", pool_size=2, num_blocks=1,
               is_residual=False, dilation_rate=(1,1)):

    def conv_unit(inp, num_filters=64, kernel_size=3, momentum=0.8, padding="same", do_act=True, dilation_rate=(1,1)):
        conv = Conv2D(num_filters, kernel_size, padding=padding, dilation_rate=dilation_rate)(inp)
        conv = BatchNormalization(momentum=momentum)(conv)
        if do_act:
            conv = LeakyReLU(0.05)(conv)

        return conv


    conv = inp
    for i in range(num_blocks):
        conv = conv_unit(conv, num_filters, kernel_size, momentum, padding, True, dilation_rate=dilation_rate)


    if pool_size is not None:
        pool = MaxPooling2D(pool_size=pool_size)(conv)

        return conv, pool
    else:
        return conv


def conv_block1D(inp, num_filters=64, kernel_size=3, momentum=0.8, padding="same", pool_size=2, num_blocks=1,
                 dilation_rate=1):

    def conv_unit(inp, num_filters=64, kernel_size=3, momentum=0.8, padding="same", do_act=True , dilation_rate=1):
        conv = Conv1D(num_filters, kernel_size, padding=padding, dilation_rate=dilation_rate)(inp)
        conv = BatchNormalization(momentum=momentum)(conv)
        if do_act:
            conv = LeakyReLU(0.05)(conv)

        return conv


    conv = inp
    for i in range(num_blocks):
        conv = conv_unit(conv, num_filters, kernel_size, momentum, padding, True, dilation_rate=dilation_rate)


    if pool_size is not None:
        pool = MaxPooling1D(pool_size=pool_size)(conv)

        return conv, pool
    else:
        return conv


def up_conv_block(inp, num_filters=64, kernel_size=3, momentum=0.8, padding="same", up_size=2, is_residual=False):

    def up_conv_unit(inp, num_filters=64, kernel_size=3, momentum=0.8, padding="same"):
        conv = Conv2D(num_filters, kernel_size, padding=padding)(inp)
        conv = BatchNormalization(momentum=momentum)(conv)
        conv = LeakyReLU(0.05)(conv)
        return conv

    upcov = UpSampling2D(size=up_size)(inp)
    #     att = Conv2D(1, kernel_size=1, activation='sigmoid')(inp2)
    #     inp2 = merge([inp2, concatenate([att]*num_filters)], mode='mul')
    #     upcov = concatenate([ upcov, inp2], axis=3)
    upcov = up_conv_unit(upcov, num_filters, kernel_size, momentum, padding)
    upcov = up_conv_unit(upcov, num_filters, kernel_size, momentum, padding)
    upcov = up_conv_unit(upcov, num_filters, 1, momentum, padding)
    if is_residual:
        upcov = add([upcov, inp])
        upcov = LeakyReLU(0.05)(upcov)
    return upcov


def up_conv_block_add(inp, inp2, num_filters=64, kernel_size=3, momentum=0.8, padding="same", up_size=2, num_blocks=3,
                      is_residual=False):

    def up_conv_unit(inp, num_filters=64, kernel_size=3, momentum=0.8, padding="same", do_act=True):
        conv = Conv2D(num_filters, kernel_size, padding=padding)(inp)
        conv = BatchNormalization(momentum=momentum)(conv)
        if do_act:
            conv = LeakyReLU(0.05)(conv)
        return conv



    if is_residual:
        inp = Conv2D(num_filters, 1, padding=padding)(inp)

    inp = UpSampling2D(size=up_size)(inp)
    upcov = inp
    #     att = Conv2D(1, kernel_size=1, activation='sigmoid')(inp2)
    #     inp2 = merge([inp2, concatenate([att]*num_filters)], mode='mul')
    upcov = concatenate([ upcov, inp2], axis=3)
    #     upcov = Add()([ upcov, inp2])
    upcov = SpatialDropout2D(0.25)(upcov)
    for i in range(num_blocks):
        upcov = up_conv_unit(upcov, num_filters, kernel_size, momentum, padding, True)
    # if num_blocks == 1:
    upcov = up_conv_unit(upcov, num_filters, 1, momentum, padding, False)

    if is_residual:
        upcov = add([upcov, inp])
    upcov = LeakyReLU(0.05)(upcov)

    return upcov





class UNetFull(BaseModelWrapper):

    def __init__(self, name, config, data_loader, input_shape=(None, None, 1)):
        super(UNetFull, self).__init__(config, data_loader, name)
        self.model = self.build_model(input_shape)


    def build_model(self, input_shape=(None, None, 1)):
        inputs = Input(input_shape)

        conv2, pool2 = conv_block(inputs, num_filters=32, kernel_size=3, num_blocks=2)
        conv3, pool3 = conv_block(pool2, num_filters=64, kernel_size=3, num_blocks=2)
        conv4, pool4 = conv_block(pool3, num_filters=128, kernel_size=3, num_blocks=2)
        conv5, pool5 = conv_block(pool4, num_filters=256, kernel_size=3, num_blocks=2, pool_size=4)
        #     conv6, pool5 = conv_block(conv5, num_filters=512, kernel_size=1)
        # pool5 = SpatialDropout2D(0.25)(pool5)
        conv_mid = conv_block(pool5, num_filters=512, kernel_size=3, num_blocks=2, pool_size=None)

        conv6 = up_conv_block_add(conv_mid, conv5, num_filters=256, kernel_size=3, num_blocks=2, up_size=4)
        conv7 = up_conv_block_add(conv6, conv4, num_filters=128, kernel_size=3, num_blocks=2)
        conv8 = up_conv_block_add(conv7, conv3, num_filters=128, kernel_size=3, num_blocks=2)
        conv9 = up_conv_block_add(conv8, conv2, num_filters=128, kernel_size=3, num_blocks=2)
        conv10 = Conv2D(1, (1, 1), activation='sigmoid', name='last_conv', padding='same')(conv9)

        model = Model(inputs=[inputs], outputs=[conv10], name=self.name)

        model.compile(optimizer='adam', loss='mse', metrics=['mae'], loss_weights=[1000])

        return model




class VGG16RegDual(BaseModelWrapper):
    def __init__(self, name, config, data_loader, input_shape=(None, None, 3)):
        super(VGG16RegDual, self).__init__(config, data_loader, name)
        self.model = self.build_model(input_shape)


    def build_model(self, input_shape=(None, None, 3)):
        vgg = VGG16(include_top=False, input_shape=input_shape)

        fc = Flatten()(vgg.output)
        output1 = Dense(1, name='output1')(fc)
        output2 = Dense(1, activation='sigmoid', name='output2')(fc)
        model = Model(vgg.input, [output1, output2], name=self.name)

        model.compile(optimizer=Adam(lr=0.00001), loss={'output1':'mse', 'output2':'binary_crossentropy'},
                      metrics={'output1':'mae', 'output2':'accuracy'}, loss_weights={'output1':1, 'output2': 100})

        return model



class VGG16Reg(BaseModelWrapper):
    def __init__(self, name, config, data_loader, input_shape=(None, None, 3)):
        super(VGG16Reg, self).__init__(config, data_loader, name)
        self.model = self.build_model(input_shape)


    def build_model(self, input_shape=(None, None, 3)):
        vgg = VGG16(include_top=False, input_shape=input_shape)

        fc = Flatten()(vgg.output)
        output1 = Dense(1, name='prediction', kernel_regularizer=regularizers.l2(1e-3))(fc)

        model = Model(vgg.input, output1, name=self.name)

        model.compile(optimizer=Adam(lr=0.00001), loss='mse',
                      metrics=['mae'])

        return model





class CNNLine(BaseModelWrapper):
    def __init__(self, name, config, data_loader, input_shape=(None, None, 3)):
        super(CNNLine, self).__init__(config, data_loader, name)
        self.model = self.build_model(input_shape)

    def build_model(self, input_shape=(None,None, 1)):
        inputs = Input(input_shape)

        conv2, pool2 = conv_block(inputs, num_filters=32, kernel_size=3, num_blocks=2)
        conv3, pool3 = conv_block(pool2, num_filters=64, kernel_size=3, num_blocks=2)
        conv4, pool4 = conv_block(pool3, num_filters=128, kernel_size=3, num_blocks=2)
        conv5, pool5 = conv_block(pool4, num_filters=256, kernel_size=5, num_blocks=1, pool_size=4)
        #     conv6, pool5 = conv_block(conv5, num_filters=512, kernel_size=1)
        # pool5 = SpatialDropout2D(0.25)(pool5)
        conv_mid = conv_block(pool5, num_filters=512, kernel_size=3, num_blocks=2, pool_size=None)
        conv_mid = GlobalMaxHorizontalPooling2D()(conv_mid)
        conv6 = up_conv_block_add_1D(conv_mid, conv5, num_filters=256, kernel_size=5, num_blocks=1, up_size=4)
        conv7 = up_conv_block_add_1D(conv6, conv4, num_filters=128, kernel_size=3, num_blocks=2)
        conv8 = up_conv_block_add_1D(conv7, conv3, num_filters=128, kernel_size=3, num_blocks=2)
        conv9 = up_conv_block_add_1D(conv8, conv2, num_filters=128, kernel_size=3, num_blocks=2)

        conv10 = Conv1D(1, 1, activation='sigmoid', name='last_conv', padding='same')(conv9)

        model = Model(inputs=[inputs], outputs=[conv10], name=self.name)

        model.compile(optimizer='adam', loss='mse', metrics=['mae'], loss_weights=[1000])
        return model

