import os
import pandas as pd
import json
import logging
from os.path import join as pjoin
from run import Run
from MLHelper.constants import *
from MLHelper.tools.utils import MLUtil
from MLHelper.audio.audioutils import AudioUtil
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset

class AudioDataset:
	"""
	Base class for handling audio dataset functionality including K-fold splitting,
	data loading, and preparing data chunks for processing.
	"""
	columns = [
		META_LABEL_1, META_PATIENT_ID, META_AUDIO_PATH, META_DATASET, META_DIAGNOSIS,
		META_QUALITY, META_SAMPLERATE, META_CHANNELS, META_LENGTH, META_BITS, META_HEARTCYCLES
	]

	def __init__(self):
		self.base_path = os.path.dirname(__file__)
		self.meta_file_train = None  # Path to the training metadata file
		self.meta_file_test = None   # Path to the testing metadata file
		self.chunk_list = None       # List of dataset chunks
		self.file_list = None        
		self.run = None
		self.PyTorch_Dataset_Class = None

	def set_run(self, run: Run):
		"""
		Set the configuration run and extract relevant parameters.
		
		Args:
		run (Run): Configuration and logging object.
		"""
		self.run = run
		self.kfold_splits = run.config[KFOLD_SPLITS]
		self.batchsize = run.config[BATCH_SIZE]

	def _self_asserts_for_training(self):
		"""
		Ensure all necessary attributes are set for training.
		"""
		assert self.meta_file_train is not None, "Training metadata file not set."
		assert self.meta_file_test is not None, "Testing metadata file not set."
		assert self.run is not None, "Run configuration not set."
		assert self.kfold_splits > 0, "Atleast one split needed."
		assert len(self.chunk_list) > 0, "No dataset chunks prepared."
		assert 0.0 <= self.run.config[TRAIN_FRAC] <= 1.0, "Training fraction must be between 0 and 1."

	def _get_kfold_entry(self, fold_number: int, train_list, valid_list) -> dict:
		"""
		Generate a dictionary containing indices and class balances for a specific K-fold.

		Args:
		fold_number (int): The fold number.
		train_list (DataFrame): The training subset.
		valid_list (DataFrame): The validation subset.

		Returns:
		dict: A dictionary with fold details including indices and class balances.
		"""
		fold_entry = {
			FOLD: fold_number, 
			TRAIN_INDEX: train_list.index.tolist(), 
			VALID_INDEX: valid_list.index.tolist(),
			TRAIN_CLASS_BALANCE: MLUtil.get_class_weights(train_list[self.run.config[LABEL_NAME]]),
			VALID_CLASS_BALANCE: MLUtil.get_class_weights(valid_list[self.run.config[LABEL_NAME]])
		}
		return fold_entry

	def prepare_kfold_splits(self):
		"""
		Prepare K-fold splits based on the chunk list. This method initializes K-fold data
		that can be used later to get dataloaders.
		"""
		self._self_asserts_for_training()
		self.kfold_split_data = []

		if self.kfold_splits == 1:
			# If no K-fold split is needed, use the training fraction to split the data.
			# no kfold split - Split with train_frac
			train_list = self.chunk_list.sample(frac=self.run.config[TRAIN_FRAC], random_state=SEED_VALUE)
			valid_list = self.chunk_list.drop(train_list.index)
			fold_entry = self._get_kfold_entry(fold_number=1, train_list=train_list, valid_list=valid_list)
			self.kfold_split_data.append(fold_entry)
		else:
			data_kfold_object = StratifiedGroupKFold(n_splits=self.kfold_splits, shuffle=True, random_state=SEED_VALUE)
			current_fold_number = 0
			label_list = self.chunk_list[self.run.config[LABEL_NAME]]
			name_list = self.chunk_list[META_PATIENT_ID]
			for train_index, val_index in data_kfold_object.split(self.chunk_list, label_list, name_list):
				current_fold_number += 1
				train_list = self.chunk_list.iloc[train_index]
				valid_list = self.chunk_list.iloc[val_index]
				fold_entry = self._get_kfold_entry(fold_number=current_fold_number, train_list=train_list, valid_list=valid_list)
				self.kfold_split_data.append(fold_entry)
		self.run.log(f"Prepared {self.kfold_splits} K-fold splits.", logger_name="metadata", level=logging.WARNING)
		self.run.log(f"K-fold split data: {self.kfold_split_data}", logger_name="metadata", level=logging.DEBUG)

	def get_dataloaders(self, num_split: int, Torch_Dataset_Class: Dataset) -> tuple[DataLoader, DataLoader]:
		"""
		Get training and validation dataloaders for a specified K-fold split.

		Args:
		num_split (int): The specific K-fold split number.
		Torch_Dataset_Class (Dataset): The PyTorch dataset class to be used.

		Returns:
		tuple[DataLoader, DataLoader]: A tuple containing the training and validation dataloaders.
		"""
		assert len(self.kfold_split_data) > 0, "K-fold splits not prepared."
		assert 1 <= num_split <= len(self.kfold_split_data), f"Invalid num_split {num_split}; should be between 1 and {len(self.kfold_split_data)}."
		
		current_fold = self.kfold_split_data[num_split - 1]
		assert num_split == current_fold[FOLD], "num_split does not match the fold data."
		
		train_indices = current_fold[TRAIN_INDEX]
		valid_indices = current_fold[VALID_INDEX]
		train_mode = DEMO if self.run.config[TASK_TYPE] == TASK_TYPE_DEMO else TRAINING
		valid_mode = DEMO if self.run.config[TASK_TYPE] == TASK_TYPE_DEMO else VALIDATION

		train_dataset = Torch_Dataset_Class(datalist=self.chunk_list.iloc[train_indices], run=self.run)
		train_dataset.set_mode(train_mode)
		valid_dataset = Torch_Dataset_Class(datalist=self.chunk_list.iloc[valid_indices], run=self.run)
		valid_dataset.set_mode(valid_mode)

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
		"""
		Prepare audio chunks based on the file list and configuration settings.
		"""
		assert self.file_list is not None, "File list not loaded."
		assert len(self.file_list) > 0, "File list is empty."

		seconds = self.run.config[CHUNK_DURATION]
		samplerate = self.target_samplerate
		audio_path = self.dataset_path
		self.chunk_list = AudioUtil.Loading.get_audio_chunk_list(
			datalist=self.file_list, target_sr=samplerate, duration=seconds,
			base_path=audio_path, logger=self.run.logger_dict["preprocessing"], padding_threshold=self.run.config[CHUNK_PADDING_THRESHOLD]
		)
		MLUtil.log_class_balance(data=self.chunk_list[self.run.config[LABEL_NAME]], logger=self.run.logger_dict["metadata"], \
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
			self.run.log(f"Loaded {len(self.file_list)} files from {file_path}", logger_name="training", level=logging.INFO)
		else:
			print(f"Loaded {len(self.file_list)} files from {file_path}")
		self.file_list = self.file_list.sample(frac=self.run.config[METADATA_FRAC], random_state=SEED_VALUE).reset_index(drop=True)
		if self.run is not None:
			self.run.log(f"Count after fractionating with {self.run.config[METADATA_FRAC]} : {len(self.file_list)}", logger_name="training", level=logging.WARNING)
		else:
			print(f"Count after fractionating with {self.run.config[METADATA_FRAC]} : {len(self.file_list)}")
		self.file_list[META_HEARTCYCLES] = self.file_list[META_HEARTCYCLES].apply(json.loads)
		MLUtil.log_class_balance(data=self.file_list[self.run.config[LABEL_NAME]], logger=self.run.logger_dict["metadata"], \
							extra_info="Audio files after frac", level=logging.WARNING)
		return self.file_list


class Physionet2016(AudioDataset):
	"""
	Derived class for handling the Physionet 2016 dataset specifics.
	"""
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
	"""
	Derived class for handling the Physionet 2022 dataset specifics.
	"""
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
