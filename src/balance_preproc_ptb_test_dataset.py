import pandas as pd
# import numpy as np
# import random
from tff_utils import pad_signal, save_as_csv

src = 'Datasets/'
df_test_balanced = pd.read_csv(src+'test_ptb.csv')
df_test_balanced['187'] = df_test_balanced['187'].apply(lambda x: int(x))

target_length = 200
last_index = df_test_balanced.shape[1] - 1

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
df_test_preproc['label'] = df_test_balanced['187']

save_as_csv(df=df_test_preproc, filename='Datasets/test_ptb_preproc.csv')
