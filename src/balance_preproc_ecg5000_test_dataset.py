import pandas as pd
# import numpy as np
import random
from scipy.io import arff
from tff_utils import pad_signal, save_as_csv, max_min_standardize

src = 'ECG5000/'
data = arff.loadarff(src+'ECG5000_TEST.arff')
df_test = pd.DataFrame(data[0])
df_test['target'] = df_test['target'].apply(lambda x: int(x)-1)

# REMOVE CLASS 4 AS THERE IS INSUFFICIENT DATA
indices4 = df_test.index[df_test['target'] == 4].to_list()
df_test = df_test.drop(indices4)

target_length = 200
last_index = df_test.shape[1] - 1

df_count = df_test.groupby(['target'], as_index=False)['att1'].count()
min_class = df_count['att1'].values.argmin()
min_value = df_count['att1'].values.min()

# GET BALANCED CLASSES FOR TEST DATA
all_indices = []
num_classes = df_count.shape[0]
index = df_test.index
for i in range(num_classes):
    indices = index[df_test['target'] == i]
    sampled_list = indices.tolist()
    if i != 2:  # not class 2
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
    original_signal = max_min_standardize(original_signal)  # standardize between 0 and 1
    original_signal = pad_signal(original_signal, target_length)  # pad
    # noisy_signal = add_noise(original_signal, sigma=sigma, length=target_length)
    collect.append(original_signal)

df_test_preproc = pd.DataFrame(data=collect)
df_test_preproc['label'] = df_test_balanced['target']

save_as_csv(df=df_test_preproc, filename='Datasets/test_ecg5000_preproc.csv')
