# Adapted from following sources:
# squeezing and exciting: https://towardsdatascience.com/squeeze-and-excitation-networks-9ef5e71eacd7
# resnet: https://stackoverflow.com/questions/64792460/how-to-code-a-residual-block-using-two-layers-of-a-basic-cnn-algorithm-built-wit
# convtransepose: https://medium.com/analytics-vidhya/building-a-convolutional-autoencoder-using-keras-using-conv2dtranspose-ca403c8d144e

import csv
import random
from abc import ABC
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, log_loss
from tensorflow import keras


class SparseRecall(tf.keras.metrics.Recall):

    def __init__(self, *, class_id, **kwargs):
        super().__init__(**kwargs)
        self.class_id = class_id

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.math.equal(y_true, self.class_id), dtype=tf.float32)
        y_pred = tf.cast(tf.math.equal(tf.math.argmax(y_pred, axis=1), self.class_id), dtype=tf.float32)
        return super().update_state(y_true, y_pred, sample_weight)

    def get_config(self):
        return super().get_config()


class Recall(tf.keras.metrics.Recall):
    def __init__(self, class_id=0, name='sparse_recall_0', thresholds=0.5, dtype=tf.float32):
        super().__init__(class_id=class_id, name=name, thresholds=thresholds, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_true = tf.reshape(y_true, [-1, 1])
        # y_pred = tf.reshape(y_pred, [-1, len(vocab), 1])
        y_pred = tf.math.argmax(y_pred, axis=1)
        return super().update_state(y_true, y_pred, sample_weight)


def dataframe_to_dataset(dataframe, label_col):
    """
       Create TF dataset from dataframe

       Input:
           1) dataframe: Dataframe
               Loaded dataframe with data.
           2) label_col: str
               Column name of labels in dataframe .
       Output:
           1) ds: dataset
               Shuffled TF dataset.

    """
    dataframe = dataframe.copy()
    labels = dataframe.pop(label_col)
    df = dataframe.values[:, :]
    df = np.resize(df, (df.shape[0], df.shape[1], 1))
    ds = tf.data.Dataset.from_tensor_slices((df, labels.values.tolist()))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds


def create_noise(sigma, length=200):
    """
       Generate noise.

       Input:
           1) Sigma: float
               Standard deviation for normal distribution to generate noise.
           2) length: int
               Length of data points.
       Output:
           1) noise_signal: list
               Noise_signal created.

    """
    noise_signal = np.random.normal(0.5, sigma, length)
    noise_signal = noise_signal / np.max(noise_signal)
    return noise_signal


def add_noise(original_signal, sigma, length=200):
    """
       Add noise to original signal.

       Input:
           1) original_signal: list
               Original signal to be augmented.
           2) sigma: float
               Standard deviation for normal distribution to generate noise.
           3) length: int
               Length of data points.
       Output:
           1) noisy_signal: list
               Augmented original signal.

    """
    noise_signal = create_noise(sigma, length)
    return original_signal + noise_signal - np.mean(noise_signal)


def pad_signal(original_signal, target_length=200):
    """
       Pad signal with zeros to reach target length.

       Input:
           1) original_signal: list
               Original signal to be augmented.
           2) target_length: int
               Length of data points.
       Output:
           1) original_signal: list
               Original_signal padded with zeros.

    """
    sample_length = len(original_signal)
    if target_length > sample_length:
        diff = target_length - sample_length
        original_signal.extend([0] * diff)
    else:
        original_signal = original_signal[:target_length]
    return original_signal


def max_min_standardize(signal):
    """
       Normalizes the data using max min method.

       Input:
           1) signal: list
               Signal data for processing.
       Output:
           1) stand_signal: list
               Normalized signal data.

    """
    max_val = np.max(np.abs(signal))  # max of absolute signal
    stand_signal = [(x/max_val + 1) for x in signal]
    return stand_signal


def train_val_split_indices_ECG5000(df_train, train_size, val_size):
    """
       Split indices for train and validation for ECG5000.

       Input:
           1) df_train: Dataframe
               Train dataframe containing all the signals.
           2) train_size: int
               Number of samples for training dataset.
           3) val_size: int
               Number of samples for validation dataset.
       Output:
           1) train_indices: list
               List of indices for training data to train model.
           2) val_indices: list
               List of indices for training data for model validation.

    """
    train_indices = []
    val_indices = []

    # remove class 4 as there is too little data
    indices4 = df_train.index[df_train['target'] == 4].to_list()
    df_train = df_train.drop(indices4)

    # class 0
    indices0 = df_train.index[df_train['target'] == 0].to_list()
    val_indices0 = random.sample(indices0, val_size)
    train_indices0 = list(set(indices0) - set(val_indices0))
    train_indices0 = random.sample(train_indices0, train_size)
    train_indices.extend(train_indices0)
    val_indices.extend(val_indices0)
    # class 1
    indices1 = df_train.index[df_train['target'] == 1].to_list()
    val_indices1 = random.sample(indices1, val_size)
    train_indices1 = list(set(indices1) - set(val_indices1))
    train_indices1 = random.sample(train_indices1, train_size)
    train_indices.extend(train_indices1)
    val_indices.extend(val_indices1)
    # class 2
    indices2 = df_train.index[df_train['target'] == 2].to_list()
    val_indices2 = random.sample(indices2, val_size)
    train_indices2 = list(set(indices2) - set(val_indices2))
    # train_indices2 = random.sample(train_indices2, train_size)
    train_indices.extend(train_indices2)
    val_indices.extend(val_indices2)
    # class 3
    indices3 = df_train.index[df_train['target'] == 3].to_list()
    val_indices3 = random.sample(indices3, val_size)
    train_indices3 = list(set(indices3) - set(val_indices3))
    train_indices3 = random.sample(train_indices3, train_size)
    train_indices.extend(train_indices3)
    val_indices.extend(val_indices3)

    return train_indices, val_indices


def train_val_split_indices_PTB(df_train, train_size, val_size, test_size):
    """
       Split indices for train and validation for ECG5000.

       Input:
           1) df_train: Dataframe
               Train dataframe containing all the signals.
           2) train_size: int
               Number of samples for training dataset.
           3) val_size: int
               Number of samples for validation dataset.
           4) test_size: int
               Number of samples for test dataset.
       Output:
           1) train_indices: list
               List of indices for training data to train model.
           2) val_indices: list
               List of indices for training data for model validation.
           3) test_indices: list
               List of indices for training data for model testing.

    """
    train_indices = []
    val_indices = []
    test_indices = []

    # class 0
    indices0 = df_train.index[df_train[187] == 0].to_list()
    test_indices0 = random.sample(indices0, test_size)
    train_val_indices0 = list(set(indices0) - set(test_indices0))
    val_indices0 = random.sample(train_val_indices0, val_size)
    train_indices0 = list(set(train_val_indices0) - set(val_indices0))
    train_indices0 = random.sample(train_indices0, train_size)
    train_indices.extend(train_indices0)
    val_indices.extend(val_indices0)
    test_indices.extend(test_indices0)

    # class 1
    indices1 = df_train.index[df_train[187] == 1].to_list()
    test_indices1 = random.sample(indices1, test_size)
    train_val_indices1 = list(set(indices1) - set(test_indices1))
    val_indices1 = random.sample(train_val_indices1, val_size)
    train_indices1 = list(set(indices1) - set(val_indices1))
    train_indices1 = random.sample(train_indices1, train_size)
    train_indices.extend(train_indices1)
    val_indices.extend(val_indices1)
    test_indices.extend(test_indices1)

    return train_indices, val_indices, test_indices


def save_as_csv(df, filename):
    """
       Save values in csv in float/int format.

       Input:
           1) df: Dataframe
               Final dataframe containing data.
           2) filename: str
               Name of file.
       Output:
           1) saved file

    """
    with open(filename, "w") as file:
        writer = csv.writer(file)
        writer.writerow(df.columns.to_list())
        writer.writerows(df.values.tolist())
    file.close()


def plot_confusion_matrix(cf_matrix, classes, cmap='Blues', title=None, figsize=None, save=False, save_name=None):
    """
       Visualize confusion matrix as heatmap.

       Input:
           1) cf_matrix: numpy array
               2D array containing confusion matrix values.
           2) classes: list
               List of classes.
           3) cmap: str
               Colour for heatmap.
           4) title: str
               Plot title.
           5) figsize: int
               Size of the heatmap plot.
           6) save: bool
               To save heatmap or not.
           7) save_name: str
               Name of the saved heatmap.
       Output: Heatmap

    """
    if figsize is None:
        # Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    plt.figure(figsize=figsize)
    sns.heatmap(cf_matrix, annot=True, fmt="", cmap=cmap, cbar=True, xticklabels=classes, yticklabels=classes)
    plt.ylabel('True labels')
    plt.xlabel('Predicted labels')
    plt.title(title)
    if save:
        plt.savefig('FL_plots/' + save_name)


def calculate_metrics(train_metrics, mode='train'):
    """
       Visualize confusion matrix as heatmap.

       Input:
           1) train_metrics: dictionary
               Containing metrics from training.
           2) mode: str
               Mode in training or not.
       Output:
           1) loss: float
               Loss scalar value.
           2) acc: float
               Accuracy scalar value.
           3) r_list: list
               List of recall values for each class.
           4) p_list: list
               List of precision values for each class.
           5) f1_list: list
               List of f1 values for each class.

    """
    if mode is 'train':
        tag = ''
    else:
        tag = 'val_'

    loss = train_metrics[mode][tag + 'loss']
    acc = train_metrics[mode][tag + 'categorical_accuracy']

    r0 = train_metrics[mode][tag + 'recall_0']
    r1 = train_metrics[mode][tag + 'recall_1']
    r2 = train_metrics[mode][tag + 'recall_2']
    r3 = train_metrics[mode][tag + 'recall_3']
    r4 = train_metrics[mode][tag + 'recall_4']

    p0 = train_metrics[mode][tag + 'prec_0']
    p1 = train_metrics[mode][tag + 'prec_1']
    p2 = train_metrics[mode][tag + 'prec_2']
    p3 = train_metrics[mode][tag + 'prec_3']
    p4 = train_metrics[mode][tag + 'prec_4']

    f1_0 = (2 * r0 * p0) / (r0 + p0)
    f1_1 = (2 * r1 * p1) / (r1 + p1)
    f1_2 = (2 * r2 * p2) / (r2 + p2)
    f1_3 = (2 * r3 * p3) / (r3 + p3)
    f1_4 = (2 * r4 * p4) / (r4 + p4)

    r_list = [r0, r1, r2, r3, r4]
    p_list = [p0, p1, p2, p3, p4]
    f1_list = [f1_0, f1_1, f1_2, f1_3, f1_4]

    return loss, acc, r_list, p_list, f1_list


def log_scalar_metrics(loss,
                       acc,
                       r_list,
                       p_list,
                       f1_list,
                       trainable_count,
                       layer_count,
                       time,
                       epoch):
    """
       Log scalar metrics into Tensorboard.

       Input:
           1) loss: float
               Loss scalar value.
           2) acc: float
               Accuracy scalar value.
           3) r_list: list
               List of recall values for each class.
           4) p_list: list
               List of precision values for each class.
           5) f1_list: list
               List of f1 values for each class.
           6) trainable_count: int/float
               Number of trainable parameters.
           7) layer_count: int/float
               Number of trainable layers.
           8) time: float
               Time taken for each epoch.
           9) epoch: int
               Epoch number.

       Output: Metrics logged into Tensorboard


    """
    r0, r1, r2, r3, r4 = r_list
    p0, p1, p2, p3, p4 = p_list
    f1_0, f1_1, f1_2, f1_3, f1_4 = f1_list

    tf.summary.scalar("Loss", loss, step=epoch)
    tf.summary.scalar("Accuracy", acc, step=epoch)
    tf.summary.scalar("Time Taken", time, step=epoch)

    tf.summary.scalar("Recall_0", r0, step=epoch)
    tf.summary.scalar("Recall_1", r1, step=epoch)
    tf.summary.scalar("Recall_2", r2, step=epoch)
    tf.summary.scalar("Recall_3", r3, step=epoch)
    tf.summary.scalar("Recall_4", r4, step=epoch)

    tf.summary.scalar("Precision_0", p0, step=epoch)
    tf.summary.scalar("Precision_1", p1, step=epoch)
    tf.summary.scalar("Precision_2", p2, step=epoch)
    tf.summary.scalar("Precision_3", p3, step=epoch)
    tf.summary.scalar("Precision_4", p4, step=epoch)

    tf.summary.scalar("F1_0", f1_0, step=epoch)
    tf.summary.scalar("F1_1", f1_1, step=epoch)
    tf.summary.scalar("F1_2", f1_2, step=epoch)
    tf.summary.scalar("F1_3", f1_3, step=epoch)
    tf.summary.scalar("F1_4", f1_4, step=epoch)

    tf.summary.scalar("Trainable_param_count", trainable_count, step=epoch)
    tf.summary.scalar("Trainable_layer_count", layer_count, step=epoch)


def get_model_with_new_last_layer(pretrained_model, num_classes=5,  activation='softmax'):
    """
       Change the last layer of the model according to the number of classes.

       Input:
           1) pretrained_model: Keras model
               Model with pre-trained weights for transfer learning.
           2) num_classes: int
               Number of classes for training dataset.
           3) activation: str
               Activation function. Default is softmax.
       Output:
           1) model: Keras model
               Modified model with new last layer.

    """
    pretrained_model.layers.pop()
    out = keras.layers.Dense(units=num_classes, activation=activation, name='last_layer')(pretrained_model.layers[-1].output)
    model = keras.Model(inputs=pretrained_model.input, outputs=out)
    return model


def freeze_all_layers(model):
    """
       Freeze all the layers of the model.

       Input:
           1) model: Keras model
               Initial model.
       Output:
           1) model: Keras model
               Model with all layers frozen.

    """
    for layer in model.layers:
        layer.trainable = False
    return model


def modify_model_for_lower_bound_loop(model):
    """
       Freeze all the layers of the model except for the fully connected layers.

       Input:
           1) model: Keras model
               Initial model.
       Output:
           1) model: Keras model
               Model with all layers frozen except for the fully connected layers.

    """
    model = freeze_all_layers(model)
    model.layers[49].trainable = True
    model.layers[51].trainable = True
    return model


def modify_model_for_upper_bound_loop(model):
    """
       Freeze all the layers of the model except for all attention and fully connected layers.

       Input:
           1) model: Keras model
               Initial model.
       Output:
           1) model: Keras model
               Model with all layers frozen except for all attention and the fully connected layers.

    """
    model = freeze_all_layers(model)
    layer_ix = [6, 7, 18, 19, 30, 31, 42, 43, 49, 51]
    for ix in layer_ix:
        model.layers[ix].trainable = True
    return model


def modify_model_for_AALF_loop(model, perm_status_dict):
    """
       Freeze all the layers of the model except for selected attention and fully connected layers.

       Input:
           1) model: Keras model
               Initial model.
            1) perm_status_dict: dict
               Dictionary with layers' indices as keys and trainable status as values.
       Output:
           1) model: Keras model
               Model with all layers frozen except for selected attention and the fully connected layers.

    """
    model = modify_model_for_lower_bound_loop(model=model)
    for ix, status in perm_status_dict.items():
        model.layers[ix].trainable = status
    return model


def check_perm_trainable_status(curr_model, prev_model, curr_permissable_list, perm_layer_frozen_period,
                                perm_status_dict, min_val, min_coverage):
    """
       Check and identify the selected attention layers to be frozen.

       Input:
           1) curr_model: keras model
               Containing current model weights.
           2) prev_model: keras model
               Containing previous model weights.
           3) curr_permissable_list: list
               List of possibly trainable attention layers that can be either frozen or unfrozen.
           4) perm_layer_frozen_period: dict
               Dictionary layers' indices as keys and number of rounds being frozen as values.
           5) perm_status_dict: dict
               Dictionary with layers' indices as keys and trainable status as values.
           6) min_val: float
               Minimum value for each weight element to be considered as significant change.
           7) min_coverage: float
               Minimum percentage of parameters in each layer acting as threshold to be frozen or not.
       Output:
           1) curr_model: keras model
               Containing current model weights.
           2) perm_layer_frozen_period: dict
               Dictionary layers' indices as keys and number of rounds being frozen as values.
           3) perm_status_dict: dict
               Dictionary with layers' indices as keys and trainable status as values.

    """
    for layer_ix in curr_permissable_list:
        # when layer is currently trainable
        if perm_status_dict[layer_ix] is True:
            # get coverage value
            coverage = get_coverage(curr_model=curr_model,
                                    prev_model=prev_model,
                                    layer_index=layer_ix,
                                    min_val=min_val)
            # check if coverage is valid
            if coverage < min_coverage:
                curr_model.layers[layer_ix].trainable = False
                perm_status_dict[layer_ix] = False
                perm_layer_frozen_period[layer_ix] = 1
        # when layer is currently not trainable
        else:
            curr_model.layers[layer_ix].trainable = True
            perm_status_dict[layer_ix] = True
            perm_layer_frozen_period[layer_ix] = 0

    return curr_model, perm_layer_frozen_period, perm_status_dict


def get_coverage(curr_model, prev_model, layer_index, min_val):
    """
       Get coverage value for each layer in the model.

       Input:
           1) curr_model: keras model
               Containing current model weights.
           2) prev_model: keras model
               Containing previous model weights.
           3) layer_index: int
               Index of layer for accessing its attributes.
           4) min_val: float
               Minimum value for each weight element to be considered as significant change.
       Output: Returns fraction of weight elements that experience significant changes.

    """
    curr_w = curr_model.layers[layer_index].weights[0].numpy()
    prev_w = prev_model.layers[layer_index].weights[0].numpy()
    w_diff = curr_w - prev_w
    mask = (w_diff >= min_val) | (w_diff <= -min_val)
    return mask.sum(axis=(0, 1)) / (w_diff.shape[0] * w_diff.shape[1])


def log_attn_and_fc_layers(model, epoch):
    """
       Log attention and fully connected layers into Tensorboard.

       Input:
           1) model: keras model
               Containing all model layers.
           2) epoch: int
               Epoch number.
       Output: Logged attention and fully connected layers into Tensorboard

    """
    layer_ix = [6, 7, 18, 19, 30, 31, 42, 43, 49, 51]
    for ix in layer_ix:
        # print(f"Num: {i}, name: {layer.name}")
        layer = model.layers[ix]
        tf.summary.histogram(layer.name + str('kernel'), layer.weights[0], step=epoch)
        tf.summary.histogram(layer.name + str('bias'), layer.weights[1], step=epoch)


def log_attn_and_fc_layers2(model, epoch):
    layer_ix = [16, 19]
    for ix in layer_ix:
        # print(f"Num: {i}, name: {layer.name}")
        layer = model.layers[ix]
        tf.summary.histogram(layer.name + str('kernel'), layer.weights[0], step=epoch)
        tf.summary.histogram(layer.name + str('bias'), layer.weights[1], step=epoch)


def calculate_val_metrics(targets, predictions, probs_preds, labels):
    """
       Calculate validation metrics.

       Input:
           1) targets: numpy array
               Containing actual labels.
           2) predictions: numpy array
               Containing predicted labels.
           3) probs_preds: numpy array
               Containing predicted probabilities.
           4) labels: list
               List of labels.
       Output:
           1) loss: float
               Loss scalar value.
           2) acc: float
               Accuracy scalar value.
           3) r_list: list
               List of recall values for each class.
           4) p_list: list
               List of precision values for each class.
           5) f1_list: list
               List of f1 values for each class.
           5) support: list
               List of sample numbers for each class.

    """
    acc = accuracy_score(targets, predictions)
    # cf = confusion_matrix(targets, predictions)
    p_list, r_list, f1_list, support = precision_recall_fscore_support(targets,
                                                                       predictions,
                                                                       average=None,
                                                                       labels=labels)
    loss = log_loss(targets, probs_preds, labels=labels)
    # loss = 0
    return loss, acc, r_list, p_list, f1_list, support


def evaluate(model, dataset):
    """
       Perform model evaluation.

       Input:
           1) model: keras model
               Containing model layers.
           2) dataset: dataset
               Containing validation data.

       Output:
           1) targets: numpy array
               Containing actual labels.
           2) predictions: numpy array
               Containing predicted labels.
           3) probs_preds: numpy array
               Containing predicted probabilities.

    """
    count = 0
    predictions = None
    targets = None
    probs_preds = None
    for batch in dataset:
        y_pred = model.predict(batch['x'])
        classes = np.argmax(y_pred, axis=1)
        true_classes = np.argmax(batch['y'].numpy(), axis=1)
        if count == 0:
            predictions = classes.copy()
            targets = true_classes.copy()
            probs_preds = y_pred.copy()
        else:
            predictions = np.append(predictions, classes)
            targets = np.append(targets, true_classes)
            probs_preds = np.concatenate((probs_preds, y_pred), axis=0)
        count += 1
    return targets, predictions, probs_preds


def _server_optimizer_fn():
    """
       Get optimizer for server.

       Input: None
       Output: keras optimizer

    """
    # tf.keras.optimizers.SGD(learning_rate=0.0001)
    # tf.keras.optimizers.Adam(learning_rate=0.0001) -> ecg500_lower
    return tf.keras.optimizers.Adam(learning_rate=0.000001)


def _client_optimizer_fn():
    """
       Get optimizer for client.

       Input: None
       Output: keras optimizer

    """
    # tf.keras.optimizers.SGD(learning_rate=0.0001)
    # tf.keras.optimizers.Adam(learning_rate=0.0001) -> ecg500_lower => 0.000001
    # 0.00001 -> ptb_lower or 0.000001
    return tf.keras.optimizers.Adam(learning_rate=0.000001)


def calculate_mean_f1_diff(curr_f1_list, prev_f1_arr):
    """
       Perform f1 scores difference calculation.

       Input:
           1) curr_f1_list: list
               Containing current f1 scores of all classes.
           2) prev_f1_arr: numpy array
               Containing previous f1 scores of all classes.

       Output:
           1) mean_f1_diff: float
               Mean score of all f1 score differences.
           2) mean_curr_f1: float
               Mean score of all current f1 scores.
           1) mean_prev_f1: float
               Mean score of all previous f1 scores.
           2) prev_f1_arr: numpy array
               Containing previous f1 scores of all classes.

    """
    curr_f1_arr = np.array(curr_f1_list)
    mean_curr_f1 = curr_f1_arr.mean()
    mean_prev_f1 = prev_f1_arr.mean()
    f1_diff_arr = curr_f1_arr - prev_f1_arr
    mean_f1_diff = f1_diff_arr.mean()
    prev_f1_arr = curr_f1_arr.copy()
    return mean_f1_diff, mean_curr_f1, mean_prev_f1, prev_f1_arr


def get_permissable_attn_layers(mean_f1_diff,
                                mean_curr_f1,
                                mean_prev_f1,
                                curr_permissable_list,
                                all_attn_list,
                                f1_diff_thresh):
    """
       Check and identify the selected attention layers to be frozen.

       Input:
           1) mean_f1_diff: mean_f1_diff: float
               Mean score of all f1 score differences.
           2) mean_curr_f1: float
               Mean score of all current f1 scores.
           3) mean_prev_f1: float
               Mean score of all previous f1 scores.
           4) curr_permissable_list: list
               List of possibly trainable attention layers that can be either frozen or unfrozen.
           5) all_attn_list: list
               List of all attention layers' indices.
           6) f1_diff_thresh: float
               Minimum value for each weight element to be considered as significant change.
       Output:
           1) curr_permissable_list: list
               Updated list of possibly trainable attention layers that can be either frozen or unfrozen.

    """
    permissable_num = len(curr_permissable_list)
    if np.abs(mean_f1_diff) <= f1_diff_thresh or mean_curr_f1 < mean_prev_f1:
        if permissable_num < len(all_attn_list):
            permissable_num += 1
            curr_permissable_list.append(all_attn_list[-permissable_num])
    return curr_permissable_list

# tf.nest.map_structure(
#    lambda a: [tf.summary.histogram(str(t) + str(n), a[n].numpy(), step=round) for t in range(len(client_weights))],
#    client_weights)
