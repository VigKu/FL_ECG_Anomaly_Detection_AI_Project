import pandas as pd
import numpy as np
from utils import train_val_split_indices, save_as_csv

src = 'ecg_kaggle/'
df_train = pd.read_csv(src + 'mitbih_train.csv', header=None)
df_train[187] = df_train[187].apply(lambda x: int(x))


train_indices, val_indices = train_val_split_indices(df_train, train_size=1000, val_size=100)

df_train_final = df_train.iloc[train_indices, :]
df_val_final = df_train.iloc[val_indices, :]

save_as_csv(df=df_train_final, filename='Datasets/train_mihbih.csv')
save_as_csv(df=df_val_final, filename='Datasets/val_mihbih.csv')
