# Note the original train and test sets of ECG5000 have been swapped to cater for the large number of samples.

import pandas as pd
# import numpy as np
from scipy.io import arff
from tff_utils import train_val_split_indices_ECG5000, save_as_csv

src = 'ECG5000/'
data = arff.loadarff(src+'ECG5000_TRAIN.arff')
df_train = pd.DataFrame(data[0])
df_train['target'] = df_train['target'].apply(lambda x: int(x)-1)
# df_train = pd.read_csv(src + 'mitbih_train.csv', header=None)
# df_train[187] = df_train[187].apply(lambda x: int(x))


train_indices, val_indices = train_val_split_indices_ECG5000(df_train, train_size=155, val_size=20)

df_train_final = df_train.iloc[train_indices, :]
df_val_final = df_train.iloc[val_indices, :]
df_val_final.columns = df_train.columns.to_list()

save_as_csv(df=df_train_final, filename='Datasets/train_ecg5000.csv')
save_as_csv(df=df_val_final, filename='Datasets/val_ecg5000.csv')
