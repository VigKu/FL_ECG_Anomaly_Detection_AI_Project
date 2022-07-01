import tensorflow as tf
import tensorflow_federated as tff
import keras_tuner as kt
from tff_utils import plot_confusion_matrix


cf_matrix = [[147, 10,  3,  1,  1],
 [22, 132,  3,  4,  1],
 [4,  2, 148, 8,  0],
 [5,  1,  9, 147,  0],
 [3,  0,  1,  1, 158]]

classes=['0', '1', '2', '3', '4']
plot_confusion_matrix(cf_matrix=cf_matrix,
                      classes=classes,
                      cmap='Blues',
                      title='Confusion Matrix on MIT-BIH Test set for attention model',
                      figsize=None,
                      save=True,
                      save_name='cfattn.png')


