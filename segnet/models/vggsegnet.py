import os

from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import Permute
from keras.layers import Reshape
from keras.layers import UpSampling2D
from keras.layers import ZeroPadding2D
from keras.models import Model

file_path = os.path.dirname(os.path.abspath(__file__))

vgg_weights_path = file_path + '/../data/vgg16_weights_th_dim_ordering_th_kernels.h5'

def VGGSegnet(classes, input_height, input_width, vgg_level=3):
    
    img_input = Input(shape=(3,input_height,input_width))
    
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format='channels_first')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format='channels_first')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format='channels_first')(x)
    
    f1 = x
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format='channels_first')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format='channels_first')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format='channels_first')(x)
    
    f2 = x
    
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format='channels_first')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format='channels_first')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format='channels_first')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format='channels_first')(x)
    
    f3 = x
    
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format='channels_first')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format='channels_first')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format='channels_first')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format='channels_first')(x)
    
    f4 = x
    
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format='channels_first')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format='channels_first')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format='channels_first')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format='channels_first')(x)
    
    f5 = x
    
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(1000, activation='softmax', name='predictions')(x)
    
    vgg = Model(img_input, x)
    vgg.load_weights(vgg_weights_path)
    
    levels = [f1, f2, f3, f4, f5]
    
    o = levels[vgg_level]
    
    o = ZeroPadding2D((1,1), data_format='channels_first')(o)
    o = Conv2D(512, (3, 3), padding='valid', data_format='channels_first')(o)
    o = BatchNormalization()(o)
    
    o = UpSampling2D((2,2), data_format='channels_first')(o)
    o = ZeroPadding2D((1,1), data_format='channels_first')(o)
    o = Conv2D(256, (3, 3), padding='valid', data_format='channels_first')(o)
    o = BatchNormalization()(o)
    
    o = UpSampling2D((2,2), data_format='channels_first')(o)
    o = ZeroPadding2D((1,1), data_format='channels_first')(o)
    o = Conv2D(128, (3, 3), padding='valid', data_format='channels_first')(o)
    o = BatchNormalization()(o)
    
    o = UpSampling2D((2,2), data_format='channels_first')(o)
    o = ZeroPadding2D((1,1), data_format='channels_first')(o)
    o = Conv2D(64, (3, 3), padding='valid', data_format='channels_first')(o)
    o = BatchNormalization()(o)
    
    o = Conv2D(n_classes, (3, 3), padding='same', data_format='channels_first')( o )
    
    o_shape = Model(img_input, o).output_shape
    outputHeight = o_shape[2]
    outputWidth = o_shape[3]
    
    o = Reshape((-1, outputHeight * outputWidth))(o)
    o = Permute((2, 1))(o)
    o = Activation('softmax')(o)
    
    model = Model(img_input, o)
    model.outputWidth = outputWidth
    model.outputHeight = outputHeight
    
    return model
