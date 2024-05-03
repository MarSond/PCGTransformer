import os
from os.path import join as pjoin
import pandas as pd
from run import Run
from MLHelper.constants import *
from MLHelper.tools.utils import MLUtil
from MLHelper.audio.audioutils import AudioUtil
import logging
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
import json


class AudioDataset:
	
	columns = [META_LABEL_1, META_PATIENT_ID, META_AUDIO_PATH, META_DATASET, META_DIAGNOSIS, \
				META_QUALITY, META_SAMPLERATE, META_CHANNELS, META_LENGTH, META_BITS, META_HEARTCYCLES]

	def __init__(self):
		self.base_path = os.path.dirname(__file__)
		self.meta_file_train = None			# File path to the training metadata
		self.meta_file_test = None
		self.chunk_list = None # Chunks generated from the dataset file list
		self.file_list = None	
		self.run = None

	def set_run(self, run: Run):
		self.run = run
		self.kfold_splits = run.config[KFOLD_SPLITS]
		self.batchsize = self.run.config[CNN_PARAMS][BATCH_SIZE]

	def _self_asserts_for_training(self):
		assert self.meta_file_train is not None, "No meta file for training set"
		assert self.meta_file_test is not None, "No meta file for test set"
		assert self.run is not None, "Run object not set"
		assert self.kfold_splits > 0, "No kfold splits prepared"
		assert len(self.chunk_list) > 0, "No dataset chunks prepared"
		assert 0 <= self.run.config[TRAIN_FRAC] <= 1, "Train fraction must be between 0 and 1"

	def _get_kfold_entry(self, fold_number: int, train_list, valid_list) -> dict:
		fold_entry = { FOLD: -1, TRAIN_INDEX: [], VALID_INDEX: [], TRAIN_CLASS_BALANCE: {}, VALID_CLASS_BALANCE: {} }
		fold_entry[FOLD] = fold_number
		fold_entry[TRAIN_INDEX] = train_list.index.to_list()
		fold_entry[VALID_INDEX] = valid_list.index.to_list()
		fold_entry[TRAIN_CLASS_BALANCE] = MLUtil.get_class_weights(train_list[self.run.config[LABEL_NAME]])
		fold_entry[VALID_CLASS_BALANCE] = MLUtil.get_class_weights(valid_list[self.run.config[LABEL_NAME]])
		return fold_entry

	def prepare_kfold_splits(self):
		# chunks need to be created. Prepares X kfold splits indicies. Later used by get_dataloaders
		# list of dicts for every fold. A dict saves the indicies, fold number and the class balance
		self._self_asserts_for_training()
		self.kfold_split_data = []

		# do kfold splits on chunk_list
		if self.kfold_splits == 0 or self.kfold_splits == 1:
			# no kfold split - Split with train_frac
			train_list = self.chunk_list.sample(frac=self.run.config[TRAIN_FRAC], random_state=SEED_VALUE)
			valid_list = self.chunk_list.drop(train_list.index)

			fold_entry = self._get_kfold_entry(fold_number=1, train_list=train_list, valid_list=valid_list)
			
			self.kfold_split_data.append(fold_entry)
		else:
			data_kfold_object = StratifiedGroupKFold(n_splits=self.run.config[KFOLD_SPLITS], shuffle=True, random_state=SEED_VALUE)
			current_fold_number = 0
			
			label_list = self.chunk_list.get(self.run.config[LABEL_NAME])
			name_list = self.chunk_list.get(META_PATIENT_ID)	# unique identifier for each file and thus hopefully patient
			for train_index, val_index in data_kfold_object.split(self.chunk_list, label_list, name_list):
				current_fold_number += 1 # fold number starts with 1
				train_list = self.chunk_list.iloc[train_index]
				valid_list = self.chunk_list.iloc[val_index]
				fold_entry = self._get_kfold_entry(fold_number=current_fold_number, train_list=train_list, valid_list=valid_list)
				self.kfold_split_data.append(fold_entry)
		self.run.train_logger.info(f"Prepared {self.kfold_splits} kfold splits")

	def get_dataloaders(self, num_split: int, Torch_Dataset_Class: Dataset) -> tuple[DataLoader, DataLoader]:
		# get a set of training and validation dataloader. num_split is the number of prepared kfold split. 
		# num_split begins regularly with 1. Value of 0 returns the full dataset
		# takes the pytorch dataset class as input
		assert len(self.kfold_split_data) > 0, "No kfold splits prepared"
		assert num_split <= len(self.kfold_split_data), \
			f"num_split {num_split} is higher than the number of prepared kfold splits of {len(self.kfold_split_data)}"
		assert num_split >= 1, f"num_split {num_split} is lower than 0"
		self._self_asserts_for_training()

		# Get training and validation indices from kfold_split_data
		current_fold = self.kfold_split_data[num_split-1] # Numbering of Folds starts with 1
		assert num_split == current_fold[FOLD], "num_split does not match the kfold split data" 
		train_indices = current_fold["train_idx"]
		valid_indices = current_fold["valid_idx"]

		if self.run.config[TASK_TYPE] == TASK_TYPE_DEMO:
			train_mode = DEMO
			valid_mode = DEMO
		else:
			train_mode = TRAINING
			valid_mode = VALIDATION


		# Create training and validation datasets
		train_dataset = Torch_Dataset_Class(datalist=self.chunk_list.iloc[train_indices], run=self.run)
		train_dataset.set_mode(train_mode)
		valid_dataset = Torch_Dataset_Class(datalist=self.chunk_list.iloc[valid_indices], run=self.run)
		valid_dataset.set_mode(valid_mode)

		
		# Create training and validation dataloaders
		if train_dataset is None or len(train_dataset) == 0:
			trainloader = None
		else:
			trainloader = DataLoader(train_dataset, batch_size=self.batchsize, shuffle=True, drop_last=False)
		if valid_dataset is None or len(valid_dataset) == 0:
			validloader = None
		else:
			validloader = DataLoader(valid_dataset, batch_size=self.batchsize, shuffle=False, drop_last=False)

		return trainloader, validloader

	def prepare_chunks(self):
		assert self.file_list is not None, "No file list loaded"
		assert len(self.file_list) > 0, "No file list loaded. Lenght 0"

		seconds = self.run.config[CHUNK_DURATION]
		samplerate = self.target_samplerate
		audio_path = self.dataset_path
		self.chunk_list = AudioUtil.Loading.get_audio_chunk_list(datalist=self.file_list, target_sr=samplerate, duration=seconds, \
														   base_path=audio_path, logger=self.run.logger_dict["preprocessing"], padding_threshold=self.run.config[CHUNK_PADDING_THRESHOLD])
		MLUtil.log_class_balance(data=self.chunk_list[self.run.config[LABEL_NAME]], logger=self.run.train_logger, \
						   extra_info="Audio files after chunking", level=logging.WARNING)
		return self.chunk_list

	def load_file_list(self, mode=TASK_TYPE_TRAINING) -> pd.DataFrame:
		if mode == TASK_TYPE_TRAINING:
			file_path = self.meta_file_train
		elif mode == "test":
			file_path = self.meta_file_test
		else:
			raise ValueError(f"Unknown mode {mode} for loading dataset")
		self.file_list = pd.read_csv(file_path, index_col="id", encoding='utf-8')
		if self.run is not None:
			self.run.log_training(f"Loaded {len(self.file_list)} files from {file_path}", level=logging.INFO)
		else:
			print(f"Loaded {len(self.file_list)} files from {file_path}")
		self.file_list = self.file_list.sample(frac=self.run.config[METADATA_FRAC], random_state=SEED_VALUE).reset_index(drop=True)
		if self.run is not None:
			self.run.log_training(f"Count after fractionating with {self.run.config[METADATA_FRAC]} : {len(self.file_list)}", level=logging.WARNING)
		else:
			print(f"Count after fractionating with {self.run.config[METADATA_FRAC]} : {len(self.file_list)}")
		self.file_list[META_HEARTCYCLES] = self.file_list[META_HEARTCYCLES].apply(json.loads)
		MLUtil.log_class_balance(data=self.file_list[self.run.config[LABEL_NAME]], logger=self.run.train_logger, \
						    extra_info="Audio files after frac", level=logging.INFO)
		return self.file_list
	


class Physionet2016(AudioDataset):

	def __init__(self):
		super().__init__()
		self.folder_name = 'physionet2016'
		self.dataset_path = pjoin(self.base_path, self.folder_name)
		self.meta_file_train = pjoin(self.dataset_path, 'train_list.csv')
		self.meta_file_test = pjoin(self.dataset_path, 'test_list.csv')

		self.train_audio_base_folder = f"{self.dataset_path}/audiofiles/train/"
		self.train_audio_search_pattern = f"{self.dataset_path}/audiofiles/train/*/*.wav"
		self.num_classes = 2
		self.target_samplerate = 2000

	

class Physionet2022(AudioDataset):

	def __init__(self):
		super().__init__()
		self.folder_name = 'physionet2022'
		self.dataset_path = pjoin(self.base_path, self.folder_name)
		self.meta_file_train = pjoin(self.dataset_path, 'train_list.csv')
		self.meta_file_test = pjoin(self.dataset_path, 'test_list.csv')
		self.train_audio_base_folder = f"{self.dataset_path}/training_data/"
		self.train_audio_search_pattern = f"{self.dataset_path}/training_data/*.wav"
		self.num_classes = 2
		self.target_samplerate = 2000

