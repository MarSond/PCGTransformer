import os
import os.path
from os.path import join as pjoin
import datetime
import pandas as pd
import random
import string
from MLHelper.tools.utils import FileUtils
import MLHelper.tools.logging_helper as logging_helper
import logging
from MLHelper.config import Config
from MLHelper.config import setup_environment
from MLHelper.audio.audioutils import AudioUtil
from data.dataset import Dataset, Physionet2016, Physionet2022

class Run:
	# create run folder
	# load and parse metadata
	# call the classifier or trainer
	def __init__(self, config_update_dict=None) -> None:
		self.config = Config() # base config with barebones
		self.setup_config(config_update_dict)
		self.setup_run_name(config_update_dict)
		self.setup_run_results_path()
		self.setup_logger(True)
		self.save_config()
		setup_environment(self.config)
		
	def setup_run_name(self, config_update_dict: dict=None):
		"""
		Sets self.run_name_suffix and self.run_name
		"""
		if config_update_dict is not None and Config.Names.run_name_suffix in config_update_dict.keys():
			self.run_name_suffix = config_update_dict[Config.Names.run_name_suffix]													# run_name_suffix is set in config_update_dict
		else:
			self.run_name_suffix = "".join(random.choices(string.ascii_uppercase, k=4))	# random string
		self.run_name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{self.run_name_suffix}"
		self.config[Config.Names.run_name] = self.run_name

	def _get_run_results_path(self, run_name):
		"""
		Returns the result directory path of the TrainingAndEvaluation run with the provided name
		"""
		return pjoin(self.config['run_folder'], run_name)
	
	def setup_run_results_path(self):
		"""
		Creates the run results directory and sets self.run_results_path and self.feature_statistics_path accordingly
		"""
		self.run_results_path = pjoin(self.config['run_folder'], self.run_name)
		directory_status = FileUtils.safe_path_create(self.run_results_path)
		if directory_status is not True:
			raise Exception(f"Could not create run results directory {self.run_results_path}.")
		# setup specific output paths here

	def setup_logger(self, log_to_file):
		"""
		Sets self.logger_dict
		"""
		log_filename = pjoin(self.run_results_path, self.config['log_output_filename']) if log_to_file else None
		log_request_dict = {"training": logging.DEBUG, "preprocessing": logging.INFO, "metadata": logging.INFO}
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
		config_save_path = pjoin(self.run_results_path, self.config["config_filename"])
		self.log_training(f"Saving configuration to {config_save_path}", level=logging.INFO)
		self.config.save_config_dict(config_save_path)

	def load_config(self) -> Config:
		"""
		Loads and returns the Config of the TrainingAndEvaluation run with the provided name
		"""
		current_path = pjoin(Run._get_run_results_path(self.run_name), self.config["config_filename"])
		return Config(config_update_path=current_path)

	def setup_config(self, config_update_dict: dict):
		"""
		Creates config object. If project config exists, it is loaded and extended with the provided config_update_dict
		Checks config_update_dict for a path to a run config and updates the config with it
		Saves self.config._config_dict as YAML under the name constants.FILENAME_RUN_CONFIG
		"""
		# check if "project_config.yaml" exists
		project_config_path = pjoin("project_config.yaml")
		if not os.path.isfile(project_config_path):
			project_config_path = None
		self.config = Config(project_config_path)
		if config_update_dict is not None and config_update_dict.get(Config.Names.load_config_from_run_name) is not None:
			run_config_path = pjoin(self._get_run_results_path(config_update_dict[Config.Names.load_config_from_run_name]), \
						   self.config["config_filename"])
			self.config.update_config_yaml(run_config_path)
			print(f"Updating config with config from {self.config['load_config_from_run_name']}")
		if config_update_dict is not None:
			# Still apply the update dict after the run config, even if extra loaded from file
			self.config.update_config_dict(config_update_dict)
		

	def start_task(self):
		"""
		Starts the task of the TrainingAndEvaluation run
		"""
		if self.config[Config.Names.task_type] == Config.Names.task_type_training:
			task_runner = TrainTask(self)
		elif self.config[Config.Names.task_type] == Config.Names.task_type_inference:
			task_runner = InferenceTask(self)
		task_runner.start_task()

class TaskBase:
	def __init__(self, run: Run) -> None:
		self.run = run
		self.task_mode = self.run.config[Config.Names.task_type]
	
	def start_task(self):
		self.run.log_training(self.run.config.get_dict(), level=logging.ERROR)
		self.load_metadata()

	def load_metadata(self):
		assert self.run.config[Config.Names.task_type] in [Config.Names.task_type_training, Config.Names.task_type_inference], \
			f"Unknown task type {self.run.config[Config.Names.task_type]}"
		if self.run.config[Config.Names.task_type] == Config.Names.task_type_training:
			dataset_name = self.run.config["train_dataset"]
		elif self.run.config[Config.Names.task_type] == Config.Names.task_type_inference:
			dataset_name = self.run.config["inference_dataset"]
		# load metadata
		self.run.log_training("Loading metadata", level=logging.INFO)
		if dataset_name == "physionet2016":
			dataset = Physionet2016()
		elif dataset_name == "physionet2022":
			dataset = Physionet2022()
		else:
			raise Exception(f"Unknown dataset name {dataset_name}")
		# load data list
		# fitler data out
		# split data into chunks
		dataset.load_dataset()
		chunk_list: pd.DataFrame = AudioUtil.Loading.get_audio_chunk_list(dataset.datalist, dataset.target_sample_rate, \
											self.run.config["chunk_duration"],\
											dataset.dataset_path, self.run.logger_dict["preprocessing"], padding_threshold=0.65)
		# load metadata
		self.run.log_training("Loaded metadata", level=logging.INFO)



class TrainTask(TaskBase):
	def __init__(self, run: Run) -> None:
		super().__init__(run)

	def start_task(self):
		print("Start training pipeline")	
		super().start_task()

class InferenceTask(TaskBase):
	def __init__(self, run: Run) -> None:
		super().__init__(run)

	def start_task(self):
		print("Start inference pipeline")
		super().start_task()	
		


	