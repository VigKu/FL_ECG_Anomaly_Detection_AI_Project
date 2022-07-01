# Adapted from following sources:
# squeezing and exciting: https://towardsdatascience.com/squeeze-and-excitation-networks-9ef5e71eacd7
# resnet: https://stackoverflow.com/questions/64792460/how-to-code-a-residual-block-using-two-layers-of-a-basic-cnn-algorithm-built-wit
# convtransepose: https://medium.com/analytics-vidhya/building-a-convolutional-autoencoder-using-keras-using-conv2dtranspose-ca403c8d144e

# from abc import ABC
import tensorflow as tf
# from tensorflow import keras
from tensorflow.keras import layers, regularizers


# import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# physical_devices = tf.config.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_devices[0], True)


def create_BaseClassifier(x, out_channels=None, linear_channels=None, kernel_size=3):  # linear_channels=[118,5]
    x = tf.nn.relu(layers.Conv1D(out_channels[0], kernel_size, padding="same")(x))
    x = tf.keras.layers.Flatten()(x)
    x = tf.nn.relu(layers.Dense(linear_channels[0])(x))
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(linear_channels[1], activation='softmax')(x)
    return x


def create_BaseEncoder(x, out_channels=None, kernel_size=None, L2=0.00001):  # out_channels=[64,64,64], kernel_size=3
    x = layers.Conv1D(out_channels[0], kernel_size, padding="same",
                      use_bias=False, kernel_regularizer=regularizers.l2(L2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2, strides=2)(tf.nn.relu(x))

    x = layers.Conv1D(out_channels[1], kernel_size, padding="same",
                      use_bias=False, kernel_regularizer=regularizers.l2(L2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2, strides=2)(tf.nn.relu(x))

    x = layers.Conv1D(out_channels[2], kernel_size, padding="same",
                      use_bias=False, kernel_regularizer=regularizers.l2(L2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2, strides=2)(tf.nn.relu(x))
    return x


def create_ECGClassifer(num_classes=5):
    input_shape = (200, 1)  # channels_last
    input_data = tf.keras.layers.Input(shape=input_shape)
    x = create_BaseEncoder(input_data, out_channels=[64, 64, 4], kernel_size=3, L2=0.00001)
    x = create_BaseClassifier(x, out_channels=[64], linear_channels=[118, num_classes], kernel_size=3)

    model = tf.keras.models.Model(input_data, x)
    return model


# save_model_dir = 'saved_models/attn/'
# test_model = create_ECGClassifer(num_classes=5)

# out = layers.Dense(3, activation='softmax')(test_model.layers[-1].output)
# model2 = tf.keras.models.Model(inputs=test_model.input, outputs=[out])
# model2.summary()
# test_model.build(input_shape=(1, 200, 1))
# test_model.load_weights(save_model_dir + 'model-25.h5')

#
# def model(self):  # optional to get output shape in model summary
#    x = keras.Input(shape=(1, 35, 12))
#    return keras.Model(inputs=[x], outputs=self.call(x))

# model = AutoEncoder().model()
# model = ECGClassifier().model()
# print(model.summary())
# modela = ECGClassifier()
# modela.classifier.create_last_layer(3)
# model2 = modela.model()
# print(model2.summary())

# for l in modela.layers:
#    print(l.name, l.trainable)

# print(modela.encoder.resblock1.trainable)
# print(modela.encoder.resblock1.seblock.trainable)
# modela.encoder.resblock1.trainable=False
# modela.encoder.resblock1.seblock.trainable=True
# print(modela.encoder.resblock1.trainable)
# print(modela.encoder.resblock1.seblock.trainable)
# print(modela.encoder.resblock1.conv1.trainable)
