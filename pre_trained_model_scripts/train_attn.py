
# https://keras.io/examples/structured_data/structured_data_classification_from_scratch/

import os
import tensorflow as tf
# import keras_tuner as kt
# from tensorflow import keras
# import numpy as np
import pandas as pd
# import sklearn as sk
# import matplotlib

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from utils import dataframe_to_dataset
from tf_attn_model import create_ECGClassifer
# from attnmodel import ECGClassifier
# from model import ECGClassifier

LR = 0.01
BATCHSIZE = 64
save_model_dir = 'saved_models/attn2/'
EPOCHS = 100

src = 'Datasets/'
df1 = pd.read_csv(src + 'train_mihbih_preproc_sigma_0.02.csv')
df2 = pd.read_csv(src + 'train_mihbih_preproc_sigma_0.05.csv')
df3 = pd.read_csv(src + 'train_mihbih_preproc_sigma_0.csv')
df_train = pd.concat([df1, df2, df3])
df_train.reset_index(drop=True, inplace=True)
df_val = pd.read_csv(src + 'val_mihbih_preproc_sigma_0.csv')


ds_train = dataframe_to_dataset(df_train, label_col='label')
ds_train = ds_train.shuffle(len(ds_train), seed=5, reshuffle_each_iteration=True)
ds_train = ds_train.batch(BATCHSIZE)

ds_val = dataframe_to_dataset(df_val, label_col='label')
ds_val = ds_val.batch(BATCHSIZE)

model = create_ECGClassifer(num_classes=5)
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=LR),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
model.fit(ds_train,
          epochs=EPOCHS,
          validation_data=ds_val,
          callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_delta=0.005),
                     tf.keras.callbacks.TensorBoard(log_dir="pre_train_logs/attn2"),
                     tf.keras.callbacks.ModelCheckpoint(filepath=save_model_dir+'model-{epoch:02d}.h5',
                                                        save_weights_only=True,
                                                        monitor='val_loss',
                                                        mode='min',
                                                        save_best_only=True)])

