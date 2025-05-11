# Federated Learning in ECG Anomaly Detection

<<<<<<< HEAD
# Content:
##	1) Folders:
###     a) report/truncated_report_copy: truncated version of original report
###		b) pre_trained_model_scripts: contains codes for pre-trained model training
####			1) data_balance/balance_mihbih_train_datatset.py: balance MIH-BIH train dataset
####			2) data_balance/balance_preproc_mihbih_test_dataset.py: balance pre-processed MIH-BIH test dataset
####			3) data_preprocess/preprocess_mihbih_train_val_dataset.py: pre-process MIH-BIH train and validation dataset
####			4) data_split/split_mihbih_train_val_dataset.py: split original MIH-BIH train data into train and validation datasets
####			5) test/test_pre_trained_model.py: test script for models on test dataset
####			6) model/tf_attn_model.py: modified model with attention network architecture in functional API
####			7) model/tf_base_model.py: base model architecture in functional API
####			8) train/train.py: Re-train base model with best hyper-parameters
####			9) train/train_attn.py: Re-train modified model with attention networks with best hyper-parameters
####			10) train/train_tune.py: tune hyper-parameters for base model
####			11) train/train_tune_attn.py: tune hyper-parameters for modified model with attention networks
####			12) utils/utils.py: contain custom made python functions

###		c) src: contain all scripts for FL training
####			Pre-processing:
#####				1) data_balance/balance_ecg5000_train_datatset.py: balance ECG5000 train dataset
#####				2) data_balance/balance_preproc_ecg5000_test_dataset.py: balance pre-processed ECG5000 test dataset
#####				3) data_balance/balance_preproc_ptb_test_dataset.py: balance pre-processed PTB test dataset
#####				4) analysis/get_confusion_matrix.py: script to plot confusion matrix 
#####				5) data_preprocess/preprocess_ecg5000_train_val_dataset.py: pre-process ECG5000 train and validation dataset
#####				6) data_preprocess/preprocess_ptb_train_val_dataset.py: pre-process PTB train and validation dataset
#####				7) data_split/split_ecg5000_train_val_dataset.py: split original ECG5000 train data into train and validation datasets
#####				8) data_split/split_ptb_train_val_test_dataset.py: split original PTB train data into train, validation and test datasets
####			Models: 
#####				1) model/tf_attn_model.py: modified model with attention network architecture in functional API
#####				2) model/tf_base_model.py: base model architecture in functional API
####			Training scripts:
#####				1) train/train_FL_AALF_ecg5000.py: training script for ECG5000 with AALF algorithm
#####				2) train/train_FL_AALF_ptb.py: training script for PTB with AALF algorithm
#####				3) train/train_FL_lower_bound_loop_ecg5000.py: training script for PTB (lower bound)
#####				4) train/train_FL_lower_bound_loop_ptb.py: training script for ECG5000 (lower bound)
#####				5) train/train_FL_upper_bound_loop_ecg5000.py: training script for ECG5000(upper bound)
#####				6) train/train_FL_upper_bound_loop_ptb.py: training script for PTB (upper bound)
####			Test script:
#####				1) test/test_FL_model.py: test script for models on FL test dataset
####			Others:
#####				1) utils/utils.py & utils/tff_utils.py: contain custom made python functions
=======
Abstract:

Can there always be a generalized model in the Federated Learning context? Is there a
way to optimize model training in a Federated environment? In the line of answering these
questions, this ambitious project hypothesizes that controlling the freezing of attention layers
independently can help to enhance the training process and learn new data by shifting the
focus of existing features in the domain of ECG. Through this process, a generalized model
can be formed by just controlling the learning of the attention layers to learn new data which
is advantageous in the context of Federated Learning. A pre-trained model (pre-trained on
MIT-BIH data) with SE-Nets as attention networks are first obtained which is then trained on
two other datasets namely PTB and ECG5000. Adaptive Attention Layer Freezing (AALF)
is implemented and applied to optimally train the attention layers by monitoring the mean F1
score and customized coverage metric for each layer. The experiments conducted in this project
have not produced sufficient favourable results, leading to the rejection of the hypothesis. The
results, however, provide insightful suggestions on the capability of shifting attention layers’
focus and on other areas of the FL training process in TFF.



Major contribution of the Dissertation:

1. The project leverages on existing works for improvement that integrates 2 papers of interests.
(a) Paper 1 - Raza et al. ”Designing ECG Monitoring Healthcare System with Federated
Transfer Learning and Explainable AI” [15] : The project improves on the model
architecture from this paper.
(b) Paper 2 - Chen et al. ”Communication-Efficient Federated Learning with Adaptive
Parameter Freezing” [16] : The project improves on the existing adaptive parameter
freezing (APF) technique from this paper.
2. Based on my research, I believe that this project will be the first to focus on adaptive
freezing for attention layers in the model (technique inspired by APF). This technique
will be called as Adaptive Attention Layer Freezing (AALF).
3. The project is unique to a large extent as it explores into distinguishing a global model
and local model for client use-case with the same model architecture with AALF in the
specific medical domain.
4. The project aims to handle anomaly ECG classes in different feature space with AALF
not seen previously when pre-trained while being cost effective.
5. Previous works in FL have been accomplished with frameworks such as FATE and PySyft
(uses PyTorch at the background). Here, the project uses TFF which is still in its initial
state (Version 0) and thus not able to support fully for practical implementation except
simulation. Thus, this paper contributes to the application/research in TFF.
>>>>>>> 728715881f5a29079afd567701eaf6cc907a3140
