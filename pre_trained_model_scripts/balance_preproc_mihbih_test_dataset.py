import pandas as pd
import numpy as np
import random
from utils import add_noise, pad_signal, save_as_csv

src = 'ecg_kaggle/'
df_test = pd.read_csv(src + 'mitbih_test.csv', header=None)

target_length = 200
last_index = df_test.shape[1] - 1

df_test[last_index] = df_test[last_index].apply(lambda x: int(x))
df_count = df_test.groupby(last_index, as_index=False)[0].count()
min_class = df_count[0].values.argmin()
min_value = df_count[0].values.min()

# GET BALANCED CLASSES FOR TEST DATA
all_indices = []
num_classes = df_count.shape[0]
index = df_test.index
for i in range(num_classes):
    indices = index[df_test[last_index] == i]
    sampled_list = indices.tolist()
    if i != 3:
        sampled_list = random.sample(sampled_list, min_value)
    all_indices.extend(sampled_list)
df_test_balanced = df_test.iloc[all_indices]
df_test_balanced.reset_index(drop=True, inplace=True)

# PAD TEST DATA
collect = []
test_samples = df_test_balanced.shape[0]
num_cols = df_test_balanced.shape[1]
for ix in range(test_samples):
    original_signal = df_test_balanced.loc[ix][:-1].to_list()
    original_signal = pad_signal(original_signal, target_length)  # pad
    # noisy_signal = add_noise(original_signal, sigma=sigma, length=target_length)
    collect.append(original_signal)

df_test_preproc = pd.DataFrame(data=collect)
df_test_preproc['label'] = df_test_balanced[num_cols - 1]

save_as_csv(df=df_test_preproc, filename='Datasets/test_mihbih_preproc.csv')
