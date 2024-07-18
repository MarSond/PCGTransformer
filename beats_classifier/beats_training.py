import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from beats_classifier import beats_models
from beats_classifier.beats_dataset import BEATsDataset
from MLHelper import constants as const
from MLHelper.dataset import AudioDataset
from MLHelper.metrics import loss as loss_functions
from MLHelper.ml_loop import ML_Loop
from run import Run


class BEATsTraining(ML_Loop):
	def __init__(self, run: Run, dataset: AudioDataset):
		super().__init__(run, dataset, BEATsDataset)
		self.run = run
		self.dataset = dataset
		self.model = None
		self.criterion = None

	def prepare_kfold_run(self):
		self.beats_params = self.run.config[const.TRANSFORMER_PARAMS]

	def set_training_utilities(self, start_model, optimizer, scheduler, scaler):
		self.model = start_model
		self.optimizer = optimizer
		self.scheduler = scheduler
		self.scaler = scaler

	def start_training_task(self, start_epoch: int, start_fold: int):
		assert self.model is not None, "Model is None. Required for training"
		assert self.optimizer is not None, "Optimizer is None. Required for training"
		assert start_epoch >= 1, "Start epoch must be greater or equal to 0"
		self.kfold_loop(start_epoch=start_epoch, start_fold=start_fold)
