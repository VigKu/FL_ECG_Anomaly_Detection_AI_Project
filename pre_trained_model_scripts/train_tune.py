
# https://keras.io/examples/structured_data/structured_data_classification_from_scratch/

import os
import tensorflow as tf
import keras_tuner as kt
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from utils import dataframe_to_dataset
# from model import ECGClassifier
from tf_base_model import create_ECGClassifer


def build_model(hp):
    model = create_ECGClassifer()
    hp_optimizer = hp.Choice('optimizer', values=['sgd', 'adam'])
    hp_lr = hp.Choice('learning_rate', values=[1e-2, 5e-2, 1e-3, 5e-3, 1e-4])
    if hp_optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=hp_lr)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=hp_lr)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    return model


src = 'Datasets/'
df1 = pd.read_csv(src + 'train_mihbih_preproc_sigma_0.02.csv')
df2 = pd.read_csv(src + 'train_mihbih_preproc_sigma_0.05.csv')
df3 = pd.read_csv(src + 'train_mihbih_preproc_sigma_0.csv')
df_train = pd.concat([df1, df2, df3])
df_train.reset_index(drop=True, inplace=True)
# df_train = pd.read_csv(src + 'train_mihbih_preproc_sigma_0.05.csv')
df_val = pd.read_csv(src + 'val_mihbih_preproc_sigma_0.csv')


ds_train = dataframe_to_dataset(df_train, label_col='label')
ds_train = ds_train.shuffle(len(ds_train), seed=5, reshuffle_each_iteration=True)
ds_train = ds_train.batch(64)

ds_val = dataframe_to_dataset(df_val, label_col='label')
ds_val = ds_val.batch(64)

hp = kt.HyperParameters()
model = build_model(hp)


tuner = kt.BayesianOptimization(
    build_model,
    objective='val_loss',
    max_trials=30,
    overwrite=True,
    directory="tmp/tb2",
    project_name="hyper_base")


tuner.search(
    ds_train,
    epochs=50,
    validation_data=ds_val,
    callbacks=[tf.keras.callbacks.TensorBoard(log_dir="tmp/tb_logs2")])


best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
print(f"LR: {best_hyperparameters.get('learning_rate')} and Optimizer: {best_hyperparameters.get('optimizer')}")
