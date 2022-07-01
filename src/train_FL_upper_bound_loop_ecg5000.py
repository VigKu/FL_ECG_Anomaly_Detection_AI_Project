import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import pandas as pd
import tensorflow as tf
import collections
import numpy as np
import tensorflow_federated as tff
import time
import tensorflow.keras.backend as K
from tff_utils import log_attn_and_fc_layers, calculate_val_metrics, evaluate, \
    _server_optimizer_fn, _client_optimizer_fn, get_model_with_new_last_layer, \
    modify_model_for_upper_bound_loop

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def dataframe_to_dataset(dataframe, label_col, num_classes=5):
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
    od = collections.OrderedDict()
    od['x'] = df
    od['y'] = tf.keras.utils.to_categorical(labels.values.tolist(), num_classes=num_classes)
    ds = tf.data.Dataset.from_tensor_slices(od)
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds


def create_tf_dataset_for_client_fn(client_id,
                                    batch=1,
                                    num_clients=3,
                                    num_classes=4,
                                    label_col='label',
                                    client_id_colname='client_id'):
    # a function which takes a client_id and returns a
    # tf.data.Dataset for that client
    client_data = df_train[df_train[client_id_colname] == client_id].copy()
    client_data.drop(columns=[client_id_colname], inplace=True)
    dataset = dataframe_to_dataset(client_data, label_col=label_col, num_classes=num_classes)
    dataset = dataset.repeat(num_clients).batch(batch)
    return dataset


def _model_fn():
    keras_model_clone = tf.keras.models.clone_model(keras_model)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    return tff.learning.from_keras_model(keras_model_clone,
                                         input_spec=data_spec,
                                         loss=loss,
                                         metrics=[tf.keras.metrics.CategoricalAccuracy(),
                                                  tf.keras.metrics.Recall(class_id=0, name='recall_0'),
                                                  tf.keras.metrics.Recall(class_id=1, name='recall_1'),
                                                  tf.keras.metrics.Recall(class_id=2, name='recall_2'),
                                                  tf.keras.metrics.Recall(class_id=3, name='recall_3'),
                                                  tf.keras.metrics.Precision(class_id=0, name='prec_0'),
                                                  tf.keras.metrics.Precision(class_id=1, name='prec_1'),
                                                  tf.keras.metrics.Precision(class_id=2, name='prec_2'),
                                                  tf.keras.metrics.Precision(class_id=3, name='prec_3'),
                                                  ])


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

    p0 = train_metrics[mode][tag + 'prec_0']
    p1 = train_metrics[mode][tag + 'prec_1']
    p2 = train_metrics[mode][tag + 'prec_2']
    p3 = train_metrics[mode][tag + 'prec_3']

    r_list = [r0, r1, r2, r3]
    p_list = [p0, p1, p2, p3]
    f1_list = []
    for r, p in zip(r_list, p_list):
        denom = r + p
        if denom != 0.0:
            f1 = (2 * r * p) / denom
            f1_list.append(f1)
        else:
            f1_list.append(0)

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
    r0, r1, r2, r3 = r_list
    p0, p1, p2, p3 = p_list
    f1_0, f1_1, f1_2, f1_3 = f1_list

    tf.summary.scalar("Loss", loss, step=epoch)
    tf.summary.scalar("Accuracy", acc, step=epoch)
    tf.summary.scalar("Time Taken", time, step=epoch)

    tf.summary.scalar("Recall_0", r0, step=epoch)
    tf.summary.scalar("Recall_1", r1, step=epoch)
    tf.summary.scalar("Recall_2", r2, step=epoch)
    tf.summary.scalar("Recall_3", r3, step=epoch)

    tf.summary.scalar("Precision_0", p0, step=epoch)
    tf.summary.scalar("Precision_1", p1, step=epoch)
    tf.summary.scalar("Precision_2", p2, step=epoch)
    tf.summary.scalar("Precision_3", p3, step=epoch)

    tf.summary.scalar("F1_0", f1_0, step=epoch)
    tf.summary.scalar("F1_1", f1_1, step=epoch)
    tf.summary.scalar("F1_2", f1_2, step=epoch)
    tf.summary.scalar("F1_3", f1_3, step=epoch)

    tf.summary.scalar("Trainable_param_count", trainable_count, step=epoch)
    tf.summary.scalar("Trainable_layer_count", layer_count, step=epoch)


# create train dataset ###
client_id_colname = 'client_id'
src = 'Datasets/'
df1 = pd.read_csv(src + 'train_ecg5000_preproc_sigma_0.csv')  # read data
df1[client_id_colname] = 1
df2 = pd.read_csv(src + 'train_ecg5000_preproc_sigma_0.02.csv')
df2[client_id_colname] = 2
df3 = pd.read_csv(src + 'train_ecg5000_preproc_sigma_0.05.csv')
df3[client_id_colname] = 3

df_train = pd.concat([df1, df2, df3])  # merge respective datasets
df_train.reset_index(drop=True, inplace=True)
# print(df1.shape)
# print(df_train.shape)
# print(df_train[client_id_colname].dtype)
train_client_ids = df_train[client_id_colname].unique()
train_data = tff.simulation.ClientData.from_clients_and_fn(
    client_ids=train_client_ids,
    create_tf_dataset_for_client_fn=create_tf_dataset_for_client_fn
)

my_train_data = [
    train_data.create_tf_dataset_for_client(client)  # create train dataset for each client
    for client in train_client_ids
]
# print(type(my_train_data[0]))
# print(len(my_train_data[0]))
# print(len(my_train_data))

# prepare directories for models and logging ###
save_model_dir = 'saved_models/attn2/'
trial_dataset = train_data.create_tf_dataset_for_client(1)
data_spec = trial_dataset.element_spec
train_writer = tf.summary.create_file_writer("FL_logs/train/ecg5000_upper/")
val_writer = tf.summary.create_file_writer("FL_logs/val/ecg5000_upper/")

# create val dataset ###
df_val = pd.read_csv(src + 'val_ecg5000_preproc_sigma_0.csv')
val_dataset = dataframe_to_dataset(df_val, label_col='label')
val_dataset = val_dataset.batch(1)


FIRST_EPOCH = True
# training loop ###
for i in range(0, 100, 4):
    if FIRST_EPOCH:
        model = tf.keras.models.load_model(save_model_dir + "whole_attn_model.h5", compile=False)
        keras_model = get_model_with_new_last_layer(pretrained_model=model, num_classes=4)
        FIRST_EPOCH = False
    else:
        keras_model.load_weights("saved_models/FL/model_ecg5000_upper.h5")

    keras_model = modify_model_for_upper_bound_loop(model=keras_model)
    keras_model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.CategoricalAccuracy(),
                 tf.keras.metrics.Recall(class_id=0, name='recall_0'),
                 tf.keras.metrics.Recall(class_id=1, name='recall_1'),
                 tf.keras.metrics.Recall(class_id=2, name='recall_2'),
                 tf.keras.metrics.Recall(class_id=3, name='recall_3'),
                 tf.keras.metrics.Precision(class_id=0, name='prec_0'),
                 tf.keras.metrics.Precision(class_id=1, name='prec_1'),
                 tf.keras.metrics.Precision(class_id=2, name='prec_2'),
                 tf.keras.metrics.Precision(class_id=3, name='prec_3'),
                 ])

    # build FL iterative process
    iterative_process = tff.learning.build_federated_averaging_process(
        model_fn=_model_fn,
        server_optimizer_fn=_server_optimizer_fn,
        client_optimizer_fn=_client_optimizer_fn
    )

    # initiate eval model
    eval_model = tf.keras.models.clone_model(keras_model)

    # initiate server
    server_state = iterative_process.initialize()

    for round_num in range(4):  # num_epochs
        start_time = time.time()
        # FL training in clients
        server_state, train_metrics = iterative_process.next(server_state, my_train_data)
        diff_time = time.time() - start_time
        epoch = i + round_num
        print(f'### Epoch {epoch} : Completed training in time: {diff_time} secs')
        if epoch % 1 == 0 or epoch == 2 - 1:
            print(f'i {i}, Round {round_num}')
            server_state.model.assign_weights_to(eval_model)
            eval_model.save_weights("saved_models/FL/model_ecg5000_upper.h5")
            trainable_count = np.sum([K.count_params(w) for w in eval_model.trainable_weights])  # trained params
            layer_count = len(eval_model.trainable_weights)  # trained layers

            # get train metrics
            loss, acc, r_list, p_list, f1_list = calculate_metrics(train_metrics)

            # Evaluation
            targets, predictions, probs_preds = evaluate(eval_model, val_dataset)
            # print(targets.shape)
            # print(predictions.shape)
            val_loss, val_acc, val_r_list, val_p_list, val_f1_list, support = calculate_val_metrics(targets,
                                                                                                    predictions,
                                                                                                    probs_preds,
                                                                                                    labels=[0, 1, 2, 3])
            print(f'Train f1:{f1_list}.')
            print(f'Validation val_f1: {val_f1_list}.')
            # Logging
            with train_writer.as_default():  # to tensorboard for train

                # log train metrics
                log_scalar_metrics(loss=loss,
                                   acc=acc,
                                   r_list=r_list,
                                   p_list=p_list,
                                   f1_list=f1_list,
                                   trainable_count=trainable_count,
                                   layer_count=layer_count,
                                   time=diff_time,
                                   epoch=epoch)
                # log attn layers
                log_attn_and_fc_layers(model=eval_model, epoch=epoch)

            with val_writer.as_default():  # to tensorboard for val
                # log val metrics
                log_scalar_metrics(loss=val_loss,
                                   acc=val_acc,
                                   r_list=val_r_list,
                                   p_list=val_p_list,
                                   f1_list=val_f1_list,
                                   trainable_count=0,
                                   layer_count=0,
                                   time=0,
                                   epoch=epoch)
