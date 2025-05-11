# train ptb dataset is already balanced from splitting so no separate script for balancing is needed

import pandas as pd
# import numpy as np
from tff_utils import add_noise, pad_signal, save_as_csv, max_min_standardize

src = 'Datasets/'
df_train = pd.read_csv(src + 'train_ptb.csv')
df_val = pd.read_csv(src + 'val_ptb.csv')
# df_train[187] = df_train[187].apply(lambda x: int(x))

sigma = 0  # 0, 0.02, 0.05
target_length = 200
num_cols = df_train.shape[1]
noise_flag = False  # True for sigma=0.02,0.05, False for sigma=0

# TRAIN DATA
collect = []
train_samples = df_train.shape[0]
for ix in range(train_samples):
    original_signal = df_train.loc[ix][:-1].to_list()
    # original_signal = max_min_standardize(original_signal)  # standardize between 0 and 1
    original_signal = pad_signal(original_signal, target_length)  # pad
    if noise_flag:
        noisy_signal = add_noise(original_signal, sigma=sigma, length=target_length)  # noise
    else:
        noisy_signal = original_signal[:]
    collect.append(noisy_signal)

df_train_preproc = pd.DataFrame(data=collect)
df_train_preproc['label'] = df_train['187']
save_as_csv(df=df_train_preproc, filename=f'Datasets/train_ptb_preproc_sigma_{sigma}.csv')

# VAL DATA
collect = []
val_samples = df_val.shape[0]
for ix in range(val_samples):
    original_signal = df_val.loc[ix][:-1].to_list()
    # original_signal = max_min_standardize(original_signal)  # standardize between 0 and 1
    original_signal = pad_signal(original_signal, target_length)  # pad
    # noisy_signal = add_noise(original_signal, sigma=sigma, length=target_length)
    collect.append(original_signal)

df_val_preproc = pd.DataFrame(data=collect)
df_val_preproc['label'] = df_val['187']
save_as_csv(df=df_val_preproc, filename=f'Datasets/val_ptb_preproc_sigma_{sigma}.csv')
