import os
import os.path
from os.path import join as pjoin
import datetime
import pandas as pd
import random
import string
import dill
from MLHelper.tools.utils import FileUtils, MLUtil
import MLHelper.tools.logging_helper as logging_helper
import logging
import torch
from MLHelper.config import Config
from MLHelper import constants as const
from MLHelper.config import setup_environment
from MLHelper.audio.audioutils import AudioUtil
from data.dataset import Physionet2016, Physionet2022

class Run:
	# create run folder
	# load and parse metadata
	# call the classifier or trainer
	def __init__(self, config_update_dict=None) -> None:
		self.config = Config() # base config with barebones
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.setup_config(config_update_dict)
		self.setup_run_name(config_update_dict)
		self.setup_run_results_path()
		self.setup_logger(log_to_file=True)
		self.save_config()
		setup_environment(self.config)
		
	def setup_run_name(self, config_update_dict: dict=None):
		"""
		Sets self.run_name_suffix and self.run_name
		"""
		if config_update_dict is not None and const.RUN_NAME_SUFFIX in config_update_dict.keys():
			self.run_name_suffix = config_update_dict[const.RUN_NAME_SUFFIX]													# run_name_suffix is set in config_update_dict
		else:
			self.run_name_suffix = "".join(random.choices(string.ascii_uppercase, k=4))	# random string
		self.run_name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{self.run_name_suffix}"
		self.config[const.RUN_NAME] = self.run_name

	def _get_run_results_path(self, run_name):
		"""
		Returns the result directory path of the TrainingAndEvaluation run with the provided name
		"""
		return pjoin(self.config[const.RUN_FOLDER], run_name)
	
	def _get_run_results_path(self, run_name):
		"""
		Returns the result directory path of the TrainingAndEvaluation run with the provided name
		"""
		return pjoin(self.config[const.RUN_FOLDER], run_name)

	def setup_run_results_path(self):
		"""
		Creates the run results directory and sets self.run_results_path and self.feature_statistics_path accordingly
		"""
		self.run_results_path = self._get_run_results_path(self.run_name)
		directory_status = FileUtils.safe_path_create(self.run_results_path)
		if directory_status is not True:
			raise Exception(f"Could not create run results directory {self.run_results_path}.")
		# setup specific output paths here

	def setup_logger(self, log_to_file):
		"""
		Sets self.logger_dict
		"""
		log_filename = pjoin(self.run_results_path, self.config[const.FILENAME_LOG_OUTPUT]) if log_to_file else None
		log_request_dict = {"training": logging.DEBUG, "preprocessing": logging.INFO, \
						"metadata": logging.INFO, "tensor": logging.DEBUG, "inference": logging.INFO}
		self.logger_dict = logging_helper.get_logger_dict(logger_map=log_request_dict, sub_name=self.run_name, to_console=True, log_filename=log_filename)
		self.logger_dict["training"].info(f"Created logger dict {self.logger_dict.keys()}. Log file: {log_filename}. Sub name: {self.run_name}")

	def log(self, message, logger_name, level=logging.INFO):
		"""
		Logs the provided message with the provided level using the logger in self.logger_dict with the provided name
		"""
		assert logger_name in self.logger_dict.keys(), f"Logger {logger_name} not found"
		self.logger_dict[logger_name].log(level, message)

	def log_training(self, message, level=logging.INFO):
		"""
		Logs the provided message with the provided level using the logger in self.logger_dict with name "training"
		"""
		self.log(message, logger_name="training", level=level)

	def save_config(self):
		"""
		Saves self.config._config_dict as YAML file under the name constants.FILENAME_RUN_CONFIG
		"""
		config_save_path = pjoin(self.run_results_path, self.config[const.FILENAME_RUN_CONFIG])
		self.log_training(f"Saving configuration to {config_save_path}", level=logging.INFO)
		self.config.save_config_dict(config_save_path)

	def load_config(self) -> Config:
		"""
		Loads and returns the Config of the TrainingAndEvaluation run with the provided name
		"""
		current_path = pjoin(Run._get_run_results_path(self.run_name), self.config[const.FILENAME_RUN_CONFIG])
		return Config(config_update_path=current_path)

	def setup_config(self, config_update_dict: dict):
		"""
		Creates config object. If project config exists, it is loaded and extended with the provided config_update_dict
		Checks config_update_dict for a path to a run config and updates the config with it
		Saves self.config._config_dict as YAML under the name constants.FILENAME_RUN_CONFIG
		"""
		# check if "project_config.py" exists, then use it "project_config: dict object as confuig update
		try:
			import project_config
			update_dict = project_config.project_config
		except Exception as e:
			update_dict = None
			print(f"Could not import project_config.py. Exception: {e}")

		self.config = Config(update_dict=update_dict)
		if config_update_dict is not None and config_update_dict.get(const.LOAD_PREVIOUS_RUN_NAME) is not None:
			run_config_path = pjoin(self._get_run_results_path(config_update_dict[const.LOAD_PREVIOUS_RUN_NAME]), \
						   self.config[const.FILENAME_RUN_CONFIG])
			self.config.update_config_yaml(run_config_path)
			print(f"Updating config with config from {self.config[const.LOAD_PREVIOUS_RUN_NAME]}")
		if config_update_dict is not None:
			# Still apply the update dict after the run config, even if extra loaded from file
			self.config.update_config_dict(config_update_dict)
		

	def start_task(self):
		"""
		Starts the task of the TrainingAndEvaluation run
		"""
		if self.config[const.TASK_TYPE] == const.TASK_TYPE_TRAINING:
			task = TrainTask(self)
		elif self.config[const.TASK_TYPE] == const.TASK_TYPE_INFERENCE:
			task = InferenceTask(self)
		self.task = task
		self.task.start_task()

class TaskBase:
	def __init__(self, run: Run) -> None:
		self.run = run
		self.config = self.run.config
		self.task_mode = self.run.config[const.TASK_TYPE]
	
	def start_task(self):
		self.run.log_training(self.run.config.get_dict(), level=logging.ERROR)
		self.load_metadata()

	def _load_dataset_object(self):
		assert self.run.config[const.TASK_TYPE] in [const.TASK_TYPE_TRAINING, const.TASK_TYPE_INFERENCE], \
			f"Unknown task type {self.run.config[const.TASK_TYPE]}"
		if self.run.config[const.TASK_TYPE] == const.TASK_TYPE_TRAINING:
			dataset_name = self.run.config[const.TRAIN_DATASET]
		elif self.run.config[const.TASK_TYPE] == const.TASK_TYPE_INFERENCE:
			dataset_name = self.run.config[const.INFERENCE_DATASET]
		# load metadata
		self.run.log_training("Loading metadata", level=logging.INFO)
		if dataset_name == "physionet2016":
			dataset = Physionet2016()
		elif dataset_name == "physionet2022":
			dataset = Physionet2022()
		else:
			raise Exception(f"Unknown dataset name {dataset_name}")
		dataset.load_dataset()
		self.run.log_training("Loaded Dataset Object", level=logging.INFO)
		return dataset

	def get_inferencer(self):
		if self.config[const.MODEL_TYPE] == const.CNN:
			from cnn_classifier import cnn_inference
			return cnn_inference.CNN_Inference
		elif self.config[const.MODEL_TYPE] == const.BEATS:
			pass
		else:
			raise Exception(f"Unknown model type for get inferencer {self.config['model_type']}")

	def load_metadata(self):
		dataset = self._load_dataset_object()
		datalist = dataset.datalist.sample(frac=self.config[const.METADATA_FRAC], \
								random_state=self.config[const.SEED]).reset_index(drop=True)
		
		chunk_list: pd.DataFrame = AudioUtil.Loading.get_audio_chunk_list(datalist, dataset.target_samplerate, \
											self.config[const.CHUNK_DURATION],\
											dataset.dataset_path, self.run.logger_dict["preprocessing"], padding_threshold=0.65)
		# TODO move to Kfold level
		MLUtil.log_class_balance(data=chunk_list[self.config['label_name']], logger=self.run.logger_dict["metadata"], extra_info="After audio chunking and frac", level=logging.WARNING)
		self.class_weights = MLUtil.get_class_weights(chunk_list[self.config['label_name']])
		self.dataset = dataset
		self.run.log_training("Loaded metadata", level=logging.INFO)

	def load_model(self, path: str):
		# load model
		self.run.log_training("Loading model", level=logging.INFO)
		if not os.path.exists(path):
			raise Exception(f"Model path {path} does not exist")
		# get model type
		if self.config[const.MODEL_TYPE] == const.CNN:
			from cnn_classifier import cnn_models
			model = cnn_models.CNN_Model_1(self.run) 
			save_dict = torch.load(path, pickle_module=dill,)
			model.load_state_dict(save_dict['model_state_dict'])
			if hasattr(self, "optimizer"):
				self.optimizer.load_state_dict(save_dict['optimizer_state_dict'])
		elif self.config[const.MODEL_TYPE] == const.BEATS:
			pass
		else:
			raise Exception(f"Unknown model type {self.config['model_type']}")
		
		self.run.log_training(f"Loaded model {path}", level=logging.INFO)
		return model


class TrainTask(TaskBase):

	def __init__(self, run: Run) -> None:
		super().__init__(run)

	def start_task(self):
		print("Start training pipeline")	


class InferenceTask(TaskBase):

	def __init__(self, run: Run) -> None:
		super().__init__(run)

	def _get_model_path(self):
		# get model path
		model_selection = self.config[const.INFERENCE_MODEL]
		model_filename = const.get_model_filename(self.config[const.MODEL_TYPE], model_selection[const.EPOCHS], model_selection[const.KFOLD])
		model_path = pjoin(self.run._get_run_results_path(self.config[const.LOAD_PREVIOUS_RUN_NAME]), const.MODEL_FOLDER, model_filename)
		return model_path

	def start_task(self):
		self.run.log_training("Starting inference pipeline", level=logging.INFO)
		super().start_task()	
		# load inference model
		inference_model = super().load_model(self._get_model_path())
		inferencer_class = super().get_inferencer()
		inferencer = inferencer_class(self.run, inference_model, self.dataset.datalist)
		self.run.log_training("Loaded all needed things for inference", level=logging.WARNING)
		inferencer.start_inference()
		# start inference
		

		


	