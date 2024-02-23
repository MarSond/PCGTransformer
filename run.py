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

class Run:
	# create run folder
	# load and parse metadata
	# call the classifier or trainer
	def __init__(self, config_update_dict=None) -> None:
		self.setup_run_name(config_update_dict)
		self.setup_run_results_path()
		self.setup_logger(True)
		self.setup_config(config_update_dict)
		setup_environment(self.config)
		
	def setup_run_name(self, config_update_dict: dict=None):
		"""
		Sets self.run_name_suffix and self.run_name
		"""
		if config_update_dict is not None and "run_name_suffix" in config_update_dict.keys():
			self.run_name_suffix = config_update_dict["run_name_suffix"]													# run_name_suffix is set in config_update_dict
		else:
			self.run_name_suffix = "".join(random.choices(string.ascii_uppercase, k=5))	# random string
		self.run_name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{self.run_name_suffix}"

	def get_run_results_path(self, run_name):
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
		self.logger_dict = logging_helper.get_logger_dict(["training", "features", "metadata"], sub_name=self.run_name, to_console=True, log_filename=log_filename)
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
		current_path = pjoin(Run.get_run_results_path(self.run_name), self.config["config_filename"])
		return Config(config_update_path=current_path)

	def setup_config(self, config_update_dict):
		"""
		Sets self.config based on the provided config_update_dict as well as config["config_update_dict_path"]
		Saves self.config._config_dict as YAML under the name constants.FILENAME_RUN_CONFIG
		"""
		self.config = Config(config_update_dict)
		if self.config["load_config_from_run_name"] is not None and self.config["load_config_from_run_name"] != "":
			run_config_path = pjoin(Run.get_run_results_path(self.config["load_config_from_run_name"]), self.config["config_filename"])
			self.config.update_config_yaml(run_config_path)
			self.log_training(f"Updating config with config from {self.config['load_config_from_run_name']}", level=logging.WARNING)
		self.save_config()