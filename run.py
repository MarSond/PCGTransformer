import datetime
import logging
import random
import string
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple

import torch

from MLHelper import constants as const
from MLHelper.config import Config, setup_environment
from MLHelper.tools import logging_helper
from MLHelper.tools.utils import FileUtils, MLModelInfo, MLUtil


class Run:
	def __init__(self, config_update_dict: Optional[dict] = None) -> None:
		self.config = Config()  # Base config with barebones
		self.setup_config(config_update_dict)
		self.setup_run_name(config_update_dict)
		self.setup_run_results_path()
		self.setup_logger(log_to_file=True)
		self.save_config()
		setup_environment(self.config)
		self.device = torch.device("cuda")
		self.log_training("Run initialized.", level=logging.INFO)
		self.task = None

	def setup_run_name(self, config_update_dict: dict):
		"""
		Sets self.run_name_suffix and self.run_name based on provided dictionary
		or generates a random suffix.
		"""
		if config_update_dict is not None \
				and const.RUN_NAME_SUFFIX in config_update_dict:
			# run_name_suffix is set in config_update_dict
			self.run_name_suffix = config_update_dict[const.RUN_NAME_SUFFIX]
		else:
			self.run_name_suffix = "".join(random.choices(
				string.ascii_uppercase, k=4))  # random string
		self.run_name = \
			f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{self.run_name_suffix}"
		self.config[const.RUN_NAME] = self.run_name

	@staticmethod
	def _get_run_results_path(config, run_name):
		"""
		Returns the result directory path of the run with the provided name
		"""
		return Path(config[const.RUN_FOLDER]) / run_name

	def setup_run_results_path(self):
		"""
		Creates the run results directory and sets run_results_path and feature_statistics_path
		"""
		self.run_results_path = Run._get_run_results_path(self.config, self.run_name)
		directory_status = FileUtils.safe_path_create(self.run_results_path)
		if directory_status is not True:
			raise Exception(f"Could not create run results directory" \
							f"{self.run_results_path}.")
		# path as pure string
		self.config[const.RUN_RESULTS_PATH] = str(self.run_results_path)

	def setup_logger(self, log_to_file):
		"""Initializes logging for different modules within the application."""
		log_filename = Path(self.run_results_path) / self.config[const.FILENAME_LOG_OUTPUT] \
			if log_to_file else None
		if MLUtil.debugger_is_active():
			log_request_dict = {
				const.LOGGER_TRAINING: \
					{	logging_helper.LEVEL_CONSOLE: logging.DEBUG,  \
						logging_helper.LEVEL_FILE: logging.DEBUG},

				const.LOGGER_PREPROCESSING: \
					{	logging_helper.LEVEL_CONSOLE: logging.INFO, \
						logging_helper.LEVEL_FILE: logging.INFO},

				const.LOGGER_LOOP: \
					{	logging_helper.LEVEL_CONSOLE: logging.DEBUG, \
						logging_helper.LEVEL_FILE: logging.DEBUG},

				const.LOGGER_METADATA: \
					{	logging_helper.LEVEL_CONSOLE: logging.DEBUG,  \
						logging_helper.LEVEL_FILE: logging.DEBUG},

				const.LOGGER_TENSOR: \
					{	logging_helper.LEVEL_CONSOLE: logging.DEBUG, \
						logging_helper.LEVEL_FILE: logging.DEBUG},
			}
		else:
			log_request_dict = {
				const.LOGGER_TRAINING: \
					{	logging_helper.LEVEL_CONSOLE: logging.INFO, \
						logging_helper.LEVEL_FILE: logging.DEBUG},
				const.LOGGER_PREPROCESSING: \
					{	logging_helper.LEVEL_CONSOLE: logging.WARNING, \
						logging_helper.LEVEL_FILE: logging.INFO},
				const.LOGGER_LOOP: \
					{	logging_helper.LEVEL_CONSOLE: logging.INFO, \
						logging_helper.LEVEL_FILE: logging.INFO},
				const.LOGGER_METADATA: \
					{	logging_helper.LEVEL_CONSOLE: logging.WARNING,  \
						logging_helper.LEVEL_FILE: logging.WARNING},
				const.LOGGER_TENSOR: \
					{	logging_helper.LEVEL_CONSOLE: logging.ERROR, \
						logging_helper.LEVEL_FILE: logging.WARNING},
			}
		self.logger_dict = logging_helper.get_logger_dict(
			logger_map=log_request_dict, sub_name=self.run_name, \
				to_console=True, log_filename=log_filename)

		self.train_logger = self.logger_dict[const.LOGGER_TRAINING]
		self.train_logger.info(f"Logger initialized. Log file: {log_filename}")


	def log(self, message, name, level=logging.INFO):
		"""Generic logging function that logs a message with a specified level and logger."""
		assert name in self.logger_dict, f"Logger {name} not found"
		self.logger_dict[name].log(level, message)

	def log_training(self, message, level=logging.INFO):
		"""Specific logging function for training related logs."""
		self.log(message, name=const.LOGGER_TRAINING, level=level)

	def log_loop(self, message, level=logging.INFO):
		"""Specific logging function for loop related logs."""
		self.log(message, name=const.LOGGER_LOOP, level=level)

	def save_config(self):
		"""Saves the current configuration to a YAML file and logs the operation."""
		config_save_path = Path(self.run_results_path) / self.config[const.FILENAME_RUN_CONFIG]
		self.log_training(f"Saving configuration to {config_save_path}", level=logging.INFO)
		self.config.save_config_dict(config_save_path)

	def load_config(self) -> Config:
		"""
		Loads and returns the Config of the TrainingAndEvaluation run with the provided name
		"""
		current_path = Path(Run._get_run_results_path(self.config, self.run_name)) /  \
			self.config[const.FILENAME_RUN_CONFIG]
		return Config(project_config_path=current_path)

	def setup_config(self, config_update: dict):
		"""
		Creates config object. If project config exists, it is loaded and
			extended with the provided config_update_dict
		Checks config_update_dict for a path to a run config and updates the config with it
		Saves self.config._config_dict as YAML under the name constants.FILENAME_RUN_CONFIG
		"""
		# check if "project_config.py" exists, then use its object as config update
		try:
			import project_config
			update_dict = project_config.project_config.copy()
		except Exception as e:
			update_dict = None
			print(f"Could not import project_config.py. Exception: {e}")  # noqa: T201

		self.config = Config(update_dict=update_dict)
		if config_update is not None and \
				config_update.get(const.LOAD_PREVIOUS_RUN_NAME) is not None:
			run_config_path = \
				Path(Run._get_run_results_path(self.config, config_update[const.LOAD_PREVIOUS_RUN_NAME])) / \
				self.config[const.FILENAME_RUN_CONFIG]
			self.config.update_config_yaml(run_config_path)
			print(f"Updating config from {self.config[const.LOAD_PREVIOUS_RUN_NAME]}") # noqa: T201
		if config_update is not None:
			# Still apply the update dict after the run config, even if extra loaded from file
			self.config.update_config_dict(config_update)
			print(f"Config updated with provided dictionary") # noqa: T201

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
		return self.task.start_task()


class TaskBase(ABC):
	"""Abstract base class for tasks in the training and evaluation run."""

	def __init__(self, run: Run) -> None:
		self.run = run
		self.config = run.config
		self.dataset = None
		self.start_epoch = 1
		self.start_fold = 1

	def get_trainer(self, run: Run, dataset):
		"""Retrieves the trainer class based on the model type in configuration."""
		model_type = self.config[const.MODEL_METHOD_TYPE]
		if model_type == const.CNN:
			from cnn_classifier import cnn_training
			return cnn_training.CNNTraining(run=run, dataset=dataset)
		if model_type == const.BEATS:
			from beats_classifier import beats_training
			return beats_training.BEATsTraining(run=run, dataset=dataset)

		self.run.log_training(f"Unknown model type {model_type}", level=logging.ERROR)
		raise ValueError(f"Unknown model type {model_type}")

	def get_inferencer(self, run: Run, dataset):
		"""Retrieves the inferencer class based on the model type in configuration."""
		model_type = self.config[const.MODEL_METHOD_TYPE]
		if model_type == const.CNN:
			from cnn_classifier import cnn_inference
			return cnn_inference.CNN_Inference(run=run, dataset=dataset)
		if model_type == const.BEATS:
			raise NotImplementedError("BEATS inference not implemented")
		self.run.log_training(f"Unknown model type {model_type}", level=logging.ERROR)
		raise ValueError(f"Unknown model type {model_type}")


	def get_dataset(self):
		"""
		Configures and returns the dataset object based on the task type and dataset configuration.
		"""
		task_type = self.config[const.TASK_TYPE]
		if task_type in \
				[const.TASK_TYPE_TRAINING, const.TASK_TYPE_INFERENCE, const.TASK_TYPE_DEMO]:
			dataset_mode = self.config.get( \
				const.TRAIN_DATASET if task_type == const.TASK_TYPE_TRAINING \
				else const.INFERENCE_DATASET)
			if dataset_mode == const.PHYSIONET_2016:
				from MLHelper.dataset import Physionet2016
				dataset = Physionet2016(self.run)
			elif dataset_mode == const.PHYSIONET_2022:
				from MLHelper.dataset import Physionet2022
				dataset = Physionet2022(self.run)
			elif dataset_mode == const.PHYSIONET_2016_2022:
				from MLHelper.dataset import Physionet2016_2022
				dataset = Physionet2016_2022(self.run)
			else:
				self.run.log_training(f"Unknown dataset {dataset_mode}", level=logging.ERROR)
				raise ValueError(f"Unknown dataset {dataset_mode}")
			dataset.load_file_list()
			dataset.prepare_chunks()
			dataset.prepare_kfold_splits()
			return dataset
		raise ValueError(f"Unknown task type {task_type}")

	@staticmethod
	def create_new_model(run: Run) -> torch.nn.Module:
		"""Retrieves the trainer class based on the model type in configuration."""
		model_type = run.config[const.MODEL_METHOD_TYPE]
		run.log_training(f"Creating new model of type {model_type}.", level=logging.INFO)
		if model_type == const.CNN:
			from cnn_classifier import cnn_models
			model = cnn_models.get_model(run)
			demo_inputs = cnn_models.get_demo_input()
		elif model_type == const.BEATS:
			from beats_classifier import beats_models
			model = beats_models.get_model(run)
			demo_inputs = beats_models.get_demo_input()
		else:
			raise ValueError(f"Unknown model type {model_type}")
		MLModelInfo.print_model_summary(model, input_data=demo_inputs, \
			logger=run.logger_dict[const.LOGGER_TENSOR])
		return model

	@staticmethod
	def _get_checkpoint_path(config: dict) -> str:
		checkpoint = config.get(const.TRAINING_CHECKPOINT)
		assert checkpoint is not None, "Checkpoint not set"
		type = config[const.MODEL_METHOD_TYPE]
		model_name = const.get_model_filename(type=type, \
			epoch=checkpoint[const.EPOCH], fold=checkpoint[const.FOLD])
		return FileUtils.join(Run._get_run_results_path( \
			config=config, run_name=checkpoint[const.RUN_NAME]), const.MODEL_FOLDER, model_name)

	@staticmethod
	def get_checkpoint(config: dict) -> Tuple[int, int, str]:
		"""
		Get the checkpoint epoch, fold, and path from the configuration.
		Returns None if no checkpoint is set.
		"""
		checkpoint = config.get(const.TRAINING_CHECKPOINT)
		if checkpoint is None:
			return None, None, None
		return checkpoint[const.EPOCH], checkpoint[const.FOLD], \
			TaskBase._get_checkpoint_path(config)

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
		self.trainer_class = None
		self.start_model = None

	def prepare_training_utilities(self):
		# create blank model, optimizer, scheduler, scaler
		model = TaskBase.create_new_model(self.run)
		optimizer, scheduler, scaler = MLUtil.get_sheduler_optimizer_scaler(config=self.config, model=model)
		return model, optimizer, scheduler, scaler

	def load_model_for_training(self) -> torch.nn.Module:
		model, optimizer, scheduler, scaler = self.prepare_training_utilities()
		# check if TRAINING_CHECKPOINT is set and load that model
		checkpoint_epoch, checkpoint_fold, checkpoint_path = self.get_checkpoint(self.config)
		if checkpoint_epoch is not None:
			assert checkpoint_fold is not None, "Fold not set in checkpoint"
			assert checkpoint_path is not None, "Path not set in checkpoint"
			self.start_epoch = checkpoint_epoch + 1 # start at next epoch
			self.start_fold = checkpoint_fold
			model, optimizer, scheduler, scaler = MLUtil.load_model( \
				model=model, device=self.run.device, optimizer=optimizer, scheduler=scheduler, scaler=scaler,\
				path=checkpoint_path, logger=self.run.train_logger)
			self.run.log_training("Loaded model and utils from checkpoint.", level=logging.DEBUG)
		#model = model.to(self.run.device)

		model, optimizer, scheduler, scaler = MLUtil.ensure_device( \
			self.run.device, model, optimizer, scheduler, scaler)
		return model, optimizer, scheduler, scaler

	def setup_task(self):
		"""Set up the training task, including loading models, datasets, and other resources."""
		self.run.log_training("Setting up training task.", level=logging.DEBUG)
		self.dataset = self.get_dataset()
		self.trainer_class = self.get_trainer(run=self.run, dataset=self.dataset)
		# check if TRAINING_CHECKPOINT is set and pass it to training loop TODO

		model, optimizer, scheduler, scaler = self.load_model_for_training()
		self.start_model = model.to(self.run.device)
		self.optimizer = optimizer#.to(self.run.device)
		self.scheduler = scheduler#.to(self.run.device)
		self.scaler = scaler#.to(self.run.device)
		self.run.log_training(f"Moved model and utils to {self.run.device}", level=logging.INFO)
		self.run.log_training("Loaded all needed things for training", level=logging.WARNING)

	def start_task(self):
		"""Starts the training process."""
		self.run.log_training("Starting training pipeline.", level=logging.INFO)
		self.trainer_class.set_training_utilities(start_model=self.start_model, \
			optimizer=self.optimizer, scheduler=self.scheduler, scaler=self.scaler)
		result = \
			self.trainer_class.start_training_task(start_epoch=self.start_epoch, start_fold=self.start_fold)
		self.run.log_training("Training complete.", level=logging.CRITICAL)
		return result

class InferenceTask(TaskBase):
	"""Task class for running inference with pre-trained models."""

	def __init__(self, run: Run) -> None:
		super().__init__(run)
		self.run.config[const.KFOLD_SPLITS] = 1  # Typically, no k-fold in inference
		self.run.config[const.EPOCHS] = 1  # Typically, only one epoch in inference
		self.run.save_config()
		self.run.log_training("Initializing Inference Task.", level=logging.INFO)

	def setup_task(self):
		"""Setup for the inference task including loading the necessary model and dataset."""
		self.run.log_training("Setting up inference task.", level=logging.DEBUG)
		self.dataset = self.get_dataset()
		model = TaskBase.create_new_model(self.run)
		model, _, _, _ = MLUtil.load_model(model=model, device=self.run.device, \
			path=self._get_inference_model_path(), logger=self.run.train_logger)
		self.inference_model = MLUtil.ensure_device(self.run.device, model)
		self.inferencer_class = self.get_inferencer(run=self.run, dataset=self.dataset)
		self.run.log_training("Loaded all needed things for inference", level=logging.WARNING)

	def _get_inference_model_path(self):
		"""Get the path to the inference model based on the configuration."""
		model_selection = self.config[const.INFERENCE_MODEL]
		model_filename = const.get_model_filename(
			self.config[const.MODEL_METHOD_TYPE], model_selection[const.EPOCHS], \
			model_selection[const.FOLD])
		return FileUtils.join(Run._get_run_results_path(self.config, \
			self.config[const.LOAD_PREVIOUS_RUN_NAME]), const.MODEL_FOLDER, model_filename)

	def start_task(self):
		"""Executes the inference pipeline, using the loaded model and dataset."""
		self.run.log_training("Starting inference pipeline.", level=logging.INFO)
		result = self.inferencer_class.start_inference_task(model=self.inference_model)
		self.run.log_training("Inference complete.", level=logging.CRITICAL)
		return result
