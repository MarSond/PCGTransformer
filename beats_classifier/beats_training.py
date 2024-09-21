import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch

from MLHelper import embedding_model
from beats_classifier.beats_dataset import BEATsDataset
from MLHelper import constants as const
from MLHelper.dataset import AudioDataset
from MLHelper.ml_loop import HookManager, ML_Loop
from MLHelper.tools.utils import FileUtils, Plotting
from run import Run

# TODO extraktor: nicht roh beats sondern beats + trainiertes 128 layer über 20 epochen

class BEATsTraining(ML_Loop):


	def __init__(self, run: Run, dataset: AudioDataset):
		super().__init__(run, dataset, BEATsDataset)
		self.run = run
		self.dataset = dataset
		self.model = None
		self.criterion = None
		self.beats_params = self.run.config[const.TRANSFORMER_PARAMS]

	def set_training_utilities(self, start_model, optimizer, scheduler, scaler):
		self.model = start_model
		self.optimizer = optimizer
		self.scheduler = scheduler
		self.scaler = scaler

	def start_training_task(self, start_epoch: int, start_fold: int):
		assert self.model is not None, "Model is None. Required for training"
		assert start_epoch >= 1, "Start epoch must be greater or equal to 0"
		return self.kfold_loop(start_epoch=start_epoch, start_fold=start_fold)
