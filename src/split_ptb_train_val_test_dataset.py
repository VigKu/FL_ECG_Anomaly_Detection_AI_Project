

import pandas as pd
# import numpy as np
from tff_utils import train_val_split_indices_PTB, save_as_csv

src = 'ecg_kaggle/'
df_normal = pd.read_csv(src+'ptbdb_normal.csv', header=None)
df_normal[187] = df_normal[187].apply(lambda x: int(x))
df_abnormal = pd.read_csv(src+'ptbdb_abnormal.csv', header=None)
df_abnormal[187] = df_abnormal[187].apply(lambda x: int(x))
df_merge = pd.concat([df_normal, df_abnormal])
df_merge.reset_index(inplace=True)
df_merge.drop(columns=['index'], inplace=True)


train_indices, val_indices, test_indices = train_val_split_indices_PTB(df_merge,
                                                                       train_size=200,
                                                                       val_size=50,
                                                                       test_size=50)

df_train_final = df_merge.iloc[train_indices, :]
df_val_final = df_merge.iloc[val_indices, :]
df_test_final = df_merge.iloc[test_indices, :]
df_val_final.columns = df_merge.columns.to_list()
df_test_final.columns = df_merge.columns.to_list()

# train ptb dataset is already balanced from splitting so no separate script for balancing is needed
save_as_csv(df=df_train_final, filename='Datasets/train_ptb.csv')
save_as_csv(df=df_val_final, filename='Datasets/val_ptb.csv')
save_as_csv(df=df_test_final, filename='Datasets/test_ptb.csv')
