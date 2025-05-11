Python Files:
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