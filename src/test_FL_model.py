import os
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from tff_utils import plot_confusion_matrix, get_model_with_new_last_layer
from utils import dataframe_to_dataset
from tf_attn_model import create_ECGClassifer

# gpu
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


TEST_DATA_DIR = 'test_ptb_preproc.csv'
SAVED_MODEL_DIR = 'saved_models/FL/'
SAVED_MODEL = 'model_ptb_aalf.h5'
CLASS_NUM = 2  # 4
# CLASS_NAMES = ['N', 'R-on-T', 'PVC', 'SP']
CLASS_NAMES = ['Normal', 'Abnormal']
PLOT_TITLE = 'Confusion Matrix on PTB Test set with AALF'
PLOT_FILE_NAME = 'cf_ptbaalf.png'
BATCHSIZE = 64

src = 'Datasets/'
df_test = pd.read_csv(src + TEST_DATA_DIR)
ds_test = dataframe_to_dataset(df_test, label_col='label')
ds_test = ds_test.batch(BATCHSIZE)

# Functional API model:
model = create_ECGClassifer(num_classes=5)
test_model = get_model_with_new_last_layer(pretrained_model=model, num_classes=CLASS_NUM)
test_model.load_weights(SAVED_MODEL_DIR + SAVED_MODEL)

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

cf_matrix = confusion_matrix(targets, predictions)
# classes=['0', '1', '2', '3', '4']
plot_confusion_matrix(cf_matrix=cf_matrix,
                      classes=CLASS_NAMES,
                      cmap='Blues',
                      title=PLOT_TITLE,
                      figsize=None,
                      save=True,
                      save_name=PLOT_FILE_NAME)

print(cf_matrix)
print(classification_report(targets, predictions, target_names=CLASS_NAMES))

