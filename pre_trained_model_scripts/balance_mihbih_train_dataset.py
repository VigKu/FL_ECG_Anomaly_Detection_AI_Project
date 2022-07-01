import pandas as pd
import numpy as np
from utils import add_noise, pad_signal, save_as_csv

src = 'Datasets/'
df_train = pd.read_csv(src + 'train_mihbih.csv')
# df_train[187] = df_train[187].apply(lambda x: int(x))

target_length = df_train.shape[1]
num_cols = df_train.shape[1]

# Additionally double sample size in train data for class 3.
df_train_class3 = df_train[df_train[str(num_cols - 1)] == 3]
df_train_class3.reset_index(inplace=True, drop='index')

# Add the augmented signal to balance for class 3.
collect = []
train_samples = df_train_class3.shape[0]
for ix in range(train_samples):
    original_signal = df_train_class3.loc[ix][:-1].to_list()
    # original_signal = pad_signal(original_signal, target_length)  # pad
    noisy_signal = add_noise(original_signal, sigma=0.005, length=target_length - 1)  # noise
    collect.append(noisy_signal)

df_train_class3_updated = pd.DataFrame(data=collect)
df_train_class3_updated['187'] = df_train_class3[str(num_cols - 1)]

# combine extra data with train data
res = df_train_class3_updated.values.tolist()
res.extend(df_train.values.tolist())
df_result = pd.DataFrame(data=res)
save_as_csv(df=df_result, filename='Datasets/train_mihbih_balance.csv')
