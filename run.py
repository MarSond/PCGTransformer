from abc import abstractmethod
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
from abc import ABC, abstractmethod


class Run:
	def __init__(self, config_update_dict=None) -> None:
		self.config = Config()  # Base config with barebones
		self.setup_config(config_update_dict)
		self.setup_run_name(config_update_dict)
		self.setup_run_results_path()
		self.setup_logger(log_to_file=True)
		self.save_config()
		setup_environment(self.config)
		self.device = torch.device("cuda")
		self.log_training("Run initialized.", level=logging.INFO)

	def setup_run_name(self, config_update_dict: dict = None):
		"""Sets self.run_name_suffix and self.run_name based on provided dictionary or generates a random suffix."""
		if config_update_dict is not None and const.RUN_NAME_SUFFIX in config_update_dict.keys():
			# run_name_suffix is set in config_update_dict
			self.run_name_suffix = config_update_dict[const.RUN_NAME_SUFFIX]
		else:
			self.run_name_suffix = "".join(random.choices(
				string.ascii_uppercase, k=4))  # random string
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

	def setup_logger(self, log_to_file):
		"""Initializes logging for different modules within the application."""
		log_filename = pjoin(self.run_results_path, self.config[const.FILENAME_LOG_OUTPUT]) if log_to_file else None
		if MLUtil.debugger_is_active():
			# TODO save model
			log_request_dict = {
				const.LOGGER_TRAINING:		{logging_helper.LEVEL_CONSOLE: logging.DEBUG, 	logging_helper.LEVEL_FILE: logging.DEBUG},
				const.LOGGER_PREPROCESSING:	{logging_helper.LEVEL_CONSOLE: logging.INFO,	logging_helper.LEVEL_FILE: logging.INFO},
				const.LOGGER_LOOP:			{logging_helper.LEVEL_CONSOLE: logging.DEBUG,	logging_helper.LEVEL_FILE: logging.DEBUG},
				const.LOGGER_METADATA:		{logging_helper.LEVEL_CONSOLE: logging.DEBUG, 	logging_helper.LEVEL_FILE: logging.DEBUG},
				const.LOGGER_TENSOR:		{logging_helper.LEVEL_CONSOLE: logging.DEBUG,	logging_helper.LEVEL_FILE: logging.ERROR},
			}
		else:
			log_request_dict = {
				const.LOGGER_TRAINING:		{logging_helper.LEVEL_CONSOLE: logging.INFO, 	logging_helper.LEVEL_FILE: logging.DEBUG},
				const.LOGGER_PREPROCESSING:	{logging_helper.LEVEL_CONSOLE: logging.WARNING,	logging_helper.LEVEL_FILE: logging.INFO},
				const.LOGGER_LOOP:			{logging_helper.LEVEL_CONSOLE: logging.INFO,	logging_helper.LEVEL_FILE: logging.DEBUG},
				const.LOGGER_METADATA:		{logging_helper.LEVEL_CONSOLE: logging.INFO, 	logging_helper.LEVEL_FILE: logging.DEBUG},
				const.LOGGER_TENSOR:		{logging_helper.LEVEL_CONSOLE: logging.WARNING,	logging_helper.LEVEL_FILE: logging.ERROR},
			}
		self.logger_dict = logging_helper.get_logger_dict(
			logger_map=log_request_dict, sub_name=self.run_name, to_console=True, log_filename=log_filename)
		
		self.train_logger = self.logger_dict[const.LOGGER_TRAINING]
		self.train_logger.info(f"Logger initialized. Log file: {log_filename}")


	def log(self, message, logger_name, level=logging.INFO):
		"""Generic logging function that logs a message with a specified level and logger."""
		assert logger_name in self.logger_dict.keys(), f"Logger {logger_name} not found"
		self.logger_dict[logger_name].log(level, message)

	def log_training(self, message, level=logging.INFO):
		"""Specific logging function for training related logs."""
		self.log(message, logger_name=const.LOGGER_TRAINING, level=level)
	
	def log_loop(self, message, level=logging.INFO):
		"""Specific logging function for loop related logs."""
		self.log(message, logger_name=const.LOGGER_LOOP, level=level)

	def save_config(self):
		"""Saves the current configuration to a YAML file and logs the operation."""
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
			update_dict = project_config.project_config.copy()
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
			print(f"Config updated with provided dictionary")

	def setup_task(self):
		"""Setup the task based on the configuration."""
		task_type = self.config[const.TASK_TYPE]
		if task_type == const.TASK_TYPE_TRAINING:
			self.task = TrainTask(self)
		elif task_type == const.TASK_TYPE_INFERENCE:
			self.task = InferenceTask(self)
		elif task_type == const.TASK_TYPE_DEMO:
			self.task = DemoTask(self)
		else:
			self.log_training(f"Unknown task type {task_type}", level=logging.ERROR)
			raise ValueError(f"Unknown task type {task_type}")
		self.task.setup_task()

	def start_task(self):
		"""Starts the configured task."""
		assert hasattr(self, "task"), "Task not set up"
		self.task.start_task()


class TaskBase(ABC):
	"""Abstract base class for tasks in the training and evaluation run."""

	def __init__(self, run: Run) -> None:
		self.run = run
		self.config = run.config

	def get_trainer(self, run: Run, dataset):
		"""Retrieves the trainer class based on the model type in configuration."""
		model_type = self.config[const.MODEL_METHOD_TYPE]
		if model_type == const.CNN:
			from cnn_classifier import cnn_training
			return cnn_training.CNN_Training(run=run, dataset=dataset)
		elif model_type == const.BEATS:
			return None
		else:
			self.run.log_training(f"Unknown model type {model_type}", level=logging.ERROR)
			raise ValueError(f"Unknown model type {model_type}")

	def get_inferencer(self, run: Run, dataset):
		"""Retrieves the inferencer class based on the model type in configuration."""
		model_type = self.config[const.MODEL_METHOD_TYPE]
		if model_type == const.CNN:
			from cnn_classifier import cnn_inference
			return cnn_inference.CNN_Inference(run=run, dataset=dataset)
		elif model_type == const.BEATS:
			return None  # Placeholder for BEATS inferencer
		else:
			self.run.log_training(f"Unknown model type {model_type}", level=logging.ERROR)
			raise ValueError(f"Unknown model type {model_type}")


	def get_dataset(self):
		"""Configures and returns the dataset object based on the task type and dataset configuration."""
		task_type = self.config[const.TASK_TYPE]
		if task_type in [const.TASK_TYPE_TRAINING, const.TASK_TYPE_INFERENCE, const.TASK_TYPE_DEMO]:
			dataset_mode = self.config.get(const.TRAIN_DATASET if task_type == const.TASK_TYPE_TRAINING else const.INFERENCE_DATASET)
			if dataset_mode == const.PHYSIONET_2016:
				from data.dataset import Physionet2016
				dataset = Physionet2016()
			elif dataset_mode == const.PHYSIONET_2022:
				from data.dataset import Physionet2022
				dataset = Physionet2022()
			else:
				self.run.log_training(f"Unknown dataset {dataset_mode}", level=logging.ERROR)
				raise ValueError(f"Unknown dataset {dataset_mode}")
			dataset.set_run(self.run)
			dataset.load_file_list()
			dataset.prepare_chunks()
			dataset.prepare_kfold_splits()
			return dataset
			
	def create_new_model(self):
		"""Retrieves the trainer class based on the model type in configuration."""
		model_type = self.config[const.MODEL_METHOD_TYPE]
		if model_type == const.CNN:
			from cnn_classifier import cnn_models
			return cnn_models.get_model(self.run)
		elif model_type == const.BEATS:
			return None
		else:
			self.run.log_training(f"Unknown model type {model_type}", level=logging.ERROR)
			raise ValueError(f"Unknown model type {model_type}")

	@abstractmethod
	def setup_task(self):
		pass

	@abstractmethod
	def start_task(self):
		pass


class DemoTask(TaskBase):
	"""Task class for demonstration purposes, may have simplified operations."""

	def __init__(self, run: Run) -> None:
		super().__init__(run)
		self.run.log_training("Initializing Demo Task.", level=logging.INFO)

	def setup_task(self):
		"""Set up the task, potentially loading needed resources or preparing settings."""
		self.run.log_training("Setting up demo task.", level=logging.DEBUG)
		self.dataset = self.get_dataset()

	def start_task(self):
		"""Starts the demo task; likely does not do much processing."""
		self.run.log_training("Starting demo task.", level=logging.INFO)


class TrainTask(TaskBase):
	"""Task class dedicated to training models."""

	def __init__(self, run: Run) -> None:
		super().__init__(run)
		self.run.log_training("Initializing Training Task.", level=logging.INFO)

	def get_model_for_training(self):
		# check if TRAINING_CHECKPOINT is set and load that model 
		model = None
		checkpoint = self.config[const.TRAINING_CHECKPOINT]
		if checkpoint is not None:
			assert const.EPOCH in checkpoint.keys(), "Epoch not set in checkpoint"
			assert const.RUN_NAME in checkpoint.keys(), "Run name not set in checkpoint"

			self.run.log_training("Loading model from checkpoint.", level=logging.DEBUG)
			model = MLUtil.load_model(path=self.config[const.TRAINING_CHECKPOINT], run=self.run, logger=self.run.train_logger)
		else:
			self.run.log_training("Creating new model.", level=logging.INFO)
			model = self.create_new_model()
		return model

	def setup_task(self):
		"""Set up the training task, including loading models, datasets, and other resources."""
		self.run.log_training("Setting up training task.", level=logging.DEBUG)
		self.dataset = self.get_dataset()
		self.trainer_class = self.get_trainer(run=self.run, dataset=self.dataset)
		# check if TRAINING_CHECKPOINT is set and pass it to training loop TODO

		self.start_model = self.get_model_for_training().to(self.run.device)
		self.run.log_training("Loaded all needed things for training", level=logging.WARNING)

	def start_task(self):
		"""Starts the training process."""
		self.run.log_training("Starting training pipeline.", level=logging.INFO)
		self.trainer_class.start_training_task(start_model=self.start_model)
		# reminder: dataset object contains all files but also the kfold splits and dataloaders prepared
		self.run.log_training("Training complete.", level=logging.CRITICAL)


class InferenceTask(TaskBase):
	"""Task class for running inference with pre-trained models."""

	def __init__(self, run: Run) -> None:
		super().__init__(run)
		self.run.config[const.KFOLD_SPLITS] = 1  # Typically, no k-fold in inference
		self.run.save_config()
		self.run.log_training("Initializing Inference Task.", level=logging.INFO)

	def setup_task(self):
		"""Setup for the inference task including loading the necessary model and dataset."""
		self.run.log_training("Setting up inference task.", level=logging.DEBUG)
		self.dataset = self.get_dataset()
		self.inference_model, _ = MLUtil.load_model(path=self._get_inference_model_path(), \
			run=self.run, logger=self.run.train_logger)
		self.inferencer_class = self.get_inferencer(run=self.run, dataset=self.dataset)
		self.run.log_training("Loaded all needed things for inference", level=logging.WARNING)

	def _get_inference_model_path(self):
		"""Get the path to the inference model based on the configuration."""
		model_selection = self.config[const.INFERENCE_MODEL]
		model_filename = const.get_model_filename(
			self.config[const.MODEL_METHOD_TYPE], model_selection[const.EPOCHS], model_selection[const.FOLD])
		model_path = pjoin(self.run._get_run_results_path(
			self.config[const.LOAD_PREVIOUS_RUN_NAME]), const.MODEL_FOLDER, model_filename)
		return model_path

	def start_task(self):
		"""Executes the inference pipeline, using the loaded model and dataset."""
		self.run.log_training("Starting inference pipeline.", level=logging.INFO)
		self.inferencer_class.start_inference_task(model=self.inference_model)
		self.run.log_training("Inference complete.", level=logging.CRITICAL)
