# Adapted from following sources:
# squeezing and exciting: https://towardsdatascience.com/squeeze-and-excitation-networks-9ef5e71eacd7
# resnet: https://stackoverflow.com/questions/64792460/how-to-code-a-residual-block-using-two-layers-of-a-basic-cnn-algorithm-built-wit
# convtransepose: https://medium.com/analytics-vidhya/building-a-convolutional-autoencoder-using-keras-using-conv2dtranspose-ca403c8d144e
import csv
import random
from abc import ABC
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


def dataframe_to_dataset(dataframe, label_col):
    """
       Create TF dataset from dataframe

       Input:
           1) dataframe: Dataframe
               Loaded dataframe with data.
           2) label_col: str
               Column name of labels in dataframe .
       Output:
           1) ds: dataset
               Shuffled TF dataset.

    """
    dataframe = dataframe.copy()
    labels = dataframe.pop(label_col)
    df = dataframe.values[:, :]
    df = np.resize(df, (df.shape[0], df.shape[1], 1))
    ds = tf.data.Dataset.from_tensor_slices((df, labels.values.tolist()))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds


def create_noise(sigma, length=200):
    """
       Generate noise.

       Input:
           1) Sigma: float
               Standard deviation for normal distribution to generate noise.
           2) length: int
               Length of data points.
       Output:
           1) noise_signal: list
               Noise_signal created.

    """
    noise_signal = np.random.normal(0.5, sigma, length)
    noise_signal = noise_signal / np.max(noise_signal)
    return noise_signal


def add_noise(original_signal, sigma, length=200):
    """
       Add noise to original signal.

       Input:
           1) original_signal: list
               Original signal to be augmented.
           2) sigma: float
               Standard deviation for normal distribution to generate noise.
           3) length: int
               Length of data points.
       Output:
           1) noisy_signal: list
               Augmented original signal.

    """
    noise_signal = create_noise(sigma, length)
    return original_signal + noise_signal - np.mean(noise_signal)


def pad_signal(original_signal, target_length=200):
    """
       Pad signal with zeros to reach target length.

       Input:
           1) original_signal: list
               Original signal to be augmented.
           2) target_length: int
               Length of data points.
       Output:
           1) original_signal: list
               Original_signal padded with zeros.

    """
    sample_length = len(original_signal)
    if target_length > sample_length:
        diff = target_length - sample_length
        original_signal.extend([0] * diff)
    else:
        original_signal = original_signal[:target_length]
    return original_signal


def train_val_split_indices(df_train, train_size, val_size):
    """
       Split indices for train and validation.

       Input:
           1) df_train: Dataframe
               Train dataframe containing all the signals.
           2) train_size: int
               Number of samples for training dataset.
           3) val_size: int
               Number of samples for validation dataset.
       Output:
           1) train_indices: list
               List of indices for training data to train model.
           2) val_indices: int
               List of indices for training data for model validation.

    """
    train_indices = []
    val_indices = []

    # class 0
    indices0 = df_train.index[df_train[187] == 0].to_list()
    val_indices0 = random.sample(indices0, val_size)
    train_indices0 = list(set(indices0) - set(val_indices0))
    train_indices0 = random.sample(train_indices0, train_size)
    train_indices.extend(train_indices0)
    val_indices.extend(val_indices0)
    # class 1
    indices1 = df_train.index[df_train[187] == 1].to_list()
    val_indices1 = random.sample(indices1, val_size)
    train_indices1 = list(set(indices1) - set(val_indices1))
    train_indices1 = random.sample(train_indices1, train_size)
    train_indices.extend(train_indices1)
    val_indices.extend(val_indices1)
    # class 2
    indices2 = df_train.index[df_train[187] == 2].to_list()
    val_indices2 = random.sample(indices2, val_size)
    train_indices2 = list(set(indices2) - set(val_indices2))
    train_indices2 = random.sample(train_indices2, train_size)
    train_indices.extend(train_indices2)
    val_indices.extend(val_indices2)
    # class 3
    indices3 = df_train.index[df_train[187] == 3].to_list()
    val_indices3 = random.sample(indices3, val_size)
    train_indices3 = list(set(indices3) - set(val_indices3))
    # train_indices3 = random.sample(train_indices3, train_size)
    train_indices.extend(train_indices3)
    val_indices.extend(val_indices3)
    # class 4
    indices4 = df_train.index[df_train[187] == 4].to_list()
    val_indices4 = random.sample(indices4, val_size)
    train_indices4 = list(set(indices4) - set(val_indices4))
    train_indices4 = random.sample(train_indices4, train_size)
    train_indices.extend(train_indices4)
    val_indices.extend(val_indices4)

    return train_indices, val_indices


def save_as_csv(df, filename):
    """
       Save values in csv in float/int format.

       Input:
           1) df: Dataframe
               Final dataframe containing data.
           2) filename: str
               Name of file.
       Output:
           1) saved file

    """
    with open(filename, "w") as file:
        writer = csv.writer(file)
        writer.writerow(df.columns.to_list())
        writer.writerows(df.values.tolist())
    file.close()

