from keras.layers import Activation, Reshape, Lambda, Dot, Add, TimeDistributed
from keras.layers import Conv1D, Conv2D, Conv3D
from keras.layers import MaxPool1D
from keras import backend as K

import tensorflow as tf

import numpy as np


def non_local_block(input:np.array, mode:str, residual:bool) -> np.array : 
    channel_dim = 1 if K.image_data_format() == 'channels_first' else -1
    input_shape = input.shape

    if channel_dim == 1:
        batch_size, channels, time_step, img_size_w, img_size_h = input_shape
    else:
        batch_size, time_step, img_size_w, img_size_h, channels = input_shape

    conv_channel = channels // 2
    if conv_channel < 1:
        conv_channel = 1
    
    if mode == 'gaussian':
        x1 = Reshape((-1, channels))(input)
        x2 = Reshape((-1, channels))(input)
        f = Dot(axes=2)([x1, x2])
        f = Activation('softmax')(f)
    
    if mode == 'dot':
        theta = _conv(input, conv_channel)
        theta = Reshape((-1, conv_channel))(theta)

        phi = _conv(input, conv_channel)
        phi = Reshape((-1, conv_channel))(phi)

        f = Dot(axes=2)([theta, phi])
        
        f = Lambda(lambda z: (1. / float(f.shape[-1])) * z)(f)

    if mode == 'embedded gaussian':
        theta = _conv(input, conv_channel)
        theta = Reshape((-1, conv_channel))(theta)

        phi = _conv(input, conv_channel)
        phi = Reshape((-1, conv_channel))(phi)

        f = Dot(axes=2)([theta, phi])
        f = Activation('softmax')(f)
    
    g = _conv(input, conv_channel)
    g = Reshape((-1, conv_channel))(g)

    y = Dot(axes=1)([f, g])

    if channel_dim == -1:
        y = Reshape((time_step, img_size_w, img_size_h, conv_channel))(y)
    else:
        y = Reshape((conv_channel, time_step, img_size_w, img_size_h))(y)

    y = _conv(y, channels)
    
    if residual:
        y = Add()([input, y])

    return y

def _conv(input, filter_num):
    x = TimeDistributed(Conv2D(filter_num, (1, 1), padding='same', use_bias=False, kernel_initializer='he_normal'))(input)
    # x = Conv3D(filter_num, (1, 1, 1), padding='same', use_bias=False, kernel_initializer='he_normal')(input)
    return x


if __name__ == '__main__':
    
    a = np.random.random(size=(64, 5, 48, 48, 4)).astype(np.float32)
    b = non_local_block(tf.constant(a), 'embedded gaussian', True)
    print(b.shape)
    