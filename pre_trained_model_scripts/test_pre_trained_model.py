import os
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

# gpu
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


from utils import dataframe_to_dataset

BATCHSIZE = 64
USE_BASE_MODEL = True  # Toggle to use base model when True or attention model when False

src = 'Datasets/'
df_test = pd.read_csv(src + 'test_mihbih_preproc.csv')
ds_test = dataframe_to_dataset(df_test, label_col='label')
ds_test = ds_test.batch(BATCHSIZE)

if USE_BASE_MODEL:
    # Subclassed model:
    # from model import ECGClassifier
    # save_model_dir = 'saved_models/base/'
    # test_model = ECGClassifier()
    # test_model.build(input_shape=(1, 200, 1))
    # test_model.load_weights(save_model_dir+'model-24.h5')

    # Functional API model:
    from tf_base_model import create_ECGClassifer
    save_model_dir = 'saved_models/base2/'
    test_model = create_ECGClassifer(num_classes=5)
    test_model.load_weights(save_model_dir + 'model-16.h5')
else:
    # Subclassed model:
    # from attnmodel import ECGClassifier
    # save_model_dir = 'saved_models/attn/'
    # test_model = ECGClassifier()
    # test_model.build(input_shape=(1, 200, 1))
    # test_model.load_weights(save_model_dir + 'model-25.h5')

    # Functional API model:
    from tf_attn_model import create_ECGClassifer
    save_model_dir = 'saved_models/attn2/'
    test_model = create_ECGClassifer(num_classes=5)
    test_model.load_weights(save_model_dir + 'model-09.h5')

count = 0
predictions = None
targets = None
for data, label in ds_test:
    y_pred = test_model.predict(data)
    classes = np.argmax(y_pred, axis=1)
    if count == 0:
        predictions = classes.copy()
        targets = label.numpy().copy()
    else:
        predictions = np.append(predictions, classes)
        targets = np.append(targets, label.numpy())
    count += 1

print(confusion_matrix(targets, predictions))
print(classification_report(targets, predictions, target_names=['N', 'V', 'Q', 'S', 'F']))

