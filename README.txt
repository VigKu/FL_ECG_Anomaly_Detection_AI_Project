


Content:
	1) Folders:
		a) pre_trained_model_scripts: contains codes for pre-trained model training
			1) data_balance/balance_mihbih_train_datatset.py: balance MIH-BIH train dataset
			2) data_balance/balance_preproc_mihbih_test_dataset.py: balance pre-processed MIH-BIH test dataset
			3) data_preprocess/preprocess_mihbih_train_val_dataset.py: pre-process MIH-BIH train and validation dataset
			4) data_split/split_mihbih_train_val_dataset.py: split original MIH-BIH train data into train and validation datasets
			5) test/test_pre_trained_model.py: test script for models on test dataset
			6) model/tf_attn_model.py: modified model with attention network architecture in functional API
			7) model/tf_base_model.py: base model architecture in functional API
			8) train/train.py: Re-train base model with best hyper-parameters
			9) train/train_attn.py: Re-train modified model with attention networks with best hyper-parameters
			10) train/train_tune.py: tune hyper-parameters for base model
			11) train/train_tune_attn.py: tune hyper-parameters for modified model with attention networks
			12) utils/utils.py: contain custom made python functions

		b) src: contain all scripts for FL training
			Pre-processing:
				1) data_balance/balance_ecg5000_train_datatset.py: balance ECG5000 train dataset
				2) data_balance/balance_preproc_ecg5000_test_dataset.py: balance pre-processed ECG5000 test dataset
				3) data_balance/balance_preproc_ptb_test_dataset.py: balance pre-processed PTB test dataset
				4) analysis/get_confusion_matrix.py: script to plot confusion matrix 
				5) data_preprocess/preprocess_ecg5000_train_val_dataset.py: pre-process ECG5000 train and validation dataset
				6) data_preprocess/preprocess_ptb_train_val_dataset.py: pre-process PTB train and validation dataset
				7) data_split/split_ecg5000_train_val_dataset.py: split original ECG5000 train data into train and validation datasets
				8) data_split/split_ptb_train_val_test_dataset.py: split original PTB train data into train, validation and test datasets
			Models: 
				1) model/tf_attn_model.py: modified model with attention network architecture in functional API
				2) model/tf_base_model.py: base model architecture in functional API
			Training scripts:
				1) train/train_FL_AALF_ecg5000.py: training script for ECG5000 with AALF algorithm
				2) train/train_FL_AALF_ptb.py: training script for PTB with AALF algorithm
				3) train/train_FL_lower_bound_loop_ecg5000.py: training script for PTB (lower bound)
				4) train/train_FL_lower_bound_loop_ptb.py: training script for ECG5000 (lower bound)
				5) train/train_FL_upper_bound_loop_ecg5000.py: training script for ECG5000(upper bound)
				6) train/train_FL_upper_bound_loop_ptb.py: training script for PTB (upper bound)
			Test script:
				1) test/test_FL_model.py: test script for models on FL test dataset
			Others:
				1) utils/utils.py & utils/tff_utils.py: contain custom made python functions
