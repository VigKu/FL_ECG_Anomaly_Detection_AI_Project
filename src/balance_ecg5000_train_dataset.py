import pandas as pd
# import numpy as np
from tff_utils import add_noise, save_as_csv

src = 'Datasets/'
df_train = pd.read_csv(src + 'train_ecg5000.csv')

target_length = df_train.shape[1]
num_cols = df_train.shape[1]

# Additionally double sample size in train data for class 2.
df_train_class2 = df_train[df_train['target'] == 2]
df_train_class2.reset_index(inplace=True, drop='index')

# Add the augmented signal to balance for class 3.
collect = []
train_samples = df_train_class2.shape[0]
for ix in range(train_samples):
    original_signal = df_train_class2.loc[ix][:-1].to_list()
    # original_signal = pad_signal(original_signal, target_length)  # pad
    noisy_signal = add_noise(original_signal, sigma=0.005, length=target_length - 1)  # noise
    collect.append(noisy_signal)

df_train_class3_updated = pd.DataFrame(data=collect)
df_train_class3_updated['target'] = df_train_class2['target']

# combine extra data with train data
res = df_train_class3_updated.values.tolist()
res.extend(df_train.values.tolist())
df_result = pd.DataFrame(data=res)
df_result.columns = df_train.columns.to_list()
save_as_csv(df=df_result, filename='Datasets/train_ecg5000_balance.csv')
