# Adapted from following sources:
# squeezing and exciting: https://towardsdatascience.com/squeeze-and-excitation-networks-9ef5e71eacd7
# resnet: https://stackoverflow.com/questions/64792460/how-to-code-a-residual-block-using-two-layers-of-a-basic-cnn-algorithm-built-wit
# convtransepose: https://medium.com/analytics-vidhya/building-a-convolutional-autoencoder-using-keras-using-conv2dtranspose-ca403c8d144e

# from abc import ABC
import tensorflow as tf
# from tensorflow import keras
from tensorflow.keras import layers, regularizers


def create_SEBlock(in_block, ch, ratio=16): # linear_channels=[118,5]
    x = layers.GlobalAveragePooling1D()(in_block)
    x = layers.Dense(ch // ratio, activation='relu')(x)
    x = layers.Dense(ch, activation='sigmoid')(x)
    return layers.Multiply()([in_block, x])


def create_SEConvResidualBlock(x, ch=None, ratio=16, kernel_size=None, L2=0.00001):
    fx = layers.Conv1D(ch, kernel_size, activation='relu', padding='same', use_bias=False)(x)
    fx = layers.BatchNormalization()(fx)
    fx = layers.Conv1D(ch, kernel_size, padding='same', use_bias=False)(fx)
    fx = layers.Add()([x, fx])
    fx = create_SEBlock(fx, ch=ch, ratio=ratio)
    out = layers.Add()([x, fx])
    out = layers.ReLU()(out)
    out = layers.BatchNormalization()(out)
    return out


def create_SEClassifier(x, out_channels=None, linear_channels=None, ratio=16, kernel_size=3, L2=0.00001): # linear_channels=[118,5]
    x = create_SEConvResidualBlock(x, out_channels[0], ratio=ratio, kernel_size=kernel_size, L2=L2)
    x = layers.Flatten()(x)
    x = layers.Dense(linear_channels[0], activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(linear_channels[1], activation='softmax')(x)
    return x


def create_SEEncoder(x, out_channels=None, ratio=16, kernel_size=None, L2=0.00001): # out_channels = [64,64,64], kernel_size = 3
    x = create_SEConvResidualBlock(x, out_channels[0], ratio=ratio, kernel_size=kernel_size, L2=L2)
    x = layers.MaxPooling1D(pool_size=2, strides=2)(x)

    x = create_SEConvResidualBlock(x, out_channels[1], ratio=ratio, kernel_size=kernel_size, L2=L2)
    x = layers.MaxPooling1D(pool_size=2, strides=2)(x)

    x = create_SEConvResidualBlock(x, out_channels[2], ratio=ratio, kernel_size=kernel_size, L2=L2)
    x = layers.MaxPooling1D(pool_size=2, strides=2)(x)
    return x


def create_ECGClassifer(num_classes=5):
    input_shape = (200, 1)  # channels_last
    input_data = tf.keras.layers.Input(shape=input_shape)
    x = create_SEEncoder(input_data, out_channels=[64, 64, 64], ratio=16, kernel_size=3, L2=0.00001)
    x = create_SEClassifier(x, out_channels=[64], linear_channels=[118, num_classes], ratio=16, kernel_size=3, L2=0.00001)

    model = tf.keras.models.Model(input_data, x)
    return model

