
Python files:

		Pre-processing:
			1) balance_ecg5000_train_datatset: balance ECG5000 train dataset
			2) balance_preproc_ecg5000_test_dataset: balance pre-processed ECG5000 test dataset
			3) balance_preproc_ptb_test_dataset: balance pre-processed PTB test dataset
			4) get_confusion_matrix: script to plot confusion matrix 
			5) preprocess_ecg5000_train_val_dataset: pre-process ECG5000 train and validation dataset
			6) preprocess_ptb_train_val_dataset: pre-process PTB train and validation dataset
			7) split_ecg5000_train_val_dataset: split original ECG5000 train data into train and validation datasets
			8) split_ptb_train_val_test_dataset: split original PTB train data into train, validation and test datasets
		Models: 
			1) tf_attn_model: modified model with attention network architecture in functional API
			2) tf_base_model: base model architecture in functional API
		Training scripts:
			1) train_FL_AALF_ecg5000: training script for ECG5000 with AALF algorithm
			2) train_FL_AALF_ptb: training script for PTB with AALF algorithm
			3) train_FL_lower_bound_loop_ecg5000: training script for PTB (lower bound)
			4) train_FL_lower_bound_loop_ptb: training script for ECG5000 (lower bound)
			5) train_FL_upper_bound_loop_ecg5000: training script for ECG5000(upper bound)
			6) train_FL_upper_bound_loop_ptb: training script for PTB (upper bound)
		Test script:
			1) test_FL_model: test script for models on FL test dataset
		Others:
			1) utils & tff_utils: contain custom made python functions