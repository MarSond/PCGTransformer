from .cnn_dataset import CNN_Dataset
import torch
from torch.utils.data import DataLoader
import pandas as pd
from MLHelper.constants import *
from MLHelper.metrics.loss import FocalLoss
import torch.optim as optim
from data.dataset import AudioDataset
from torch.cuda.amp.grad_scaler import GradScaler
from run import Run
from MLHelper.ml_loop import ML_Loop
import logging


class CNN_Training(ML_Loop):

	def __init__(self, run: Run, dataset: AudioDataset) -> None:
		super().__init__(run, dataset, pytorch_dataset_class=CNN_Dataset)

	def num_worker_test(self, logger: logging.Logger):
		from time import time
		import multiprocessing as mp
		self.run_config = self.base_config
		train_list, valid_list = self._get_train_valid_list(None, None)
		dataset = CNN_Dataset(datalist=train_list, run=self.run)
		for num_workers in range(0, mp.cpu_count() + 1, 2):
			start = time()
			train_loader = DataLoader(
				dataset, shuffle=True, num_workers=num_workers, batch_size=64, drop_last=True)
			dl_it = iter(train_loader)
			for epoch in range(1, 5):
				for _, (data_batch, labels) in enumerate(dl_it):
					pass
			end = time()
			logger.error("Finish with:{} second, num_workers={}".format(
				end - start, num_workers))


	def start_training_task(self, start_model, start_epoch=0):
		self.model = start_model
		"""
		model is either a new model or a checkpointed model
		start_epoch is the epoch to start at. The loop beginns from start but skeips training untill start_epoch
		"""
		self.kfold_loop(start_epoch=start_epoch)

	def prepare_self(self):
		# model = self.get_model() # TODO get from task (new or checkpointed) TODO Log table optionally
		"""if self.logger.isEnabledFor(logging.DEBUG):
			self.logger.debug(MLUtil.get_model_table(self.model))
			#MLUtil.(self.model, (self.base_config['batchsize'], 1, 72, 157)) """
		self.cnn_params = self.run.config[CNN_PARAMS]
		if self.cnn_params[OPTIMIZER] == OPTIMIZER_ADAM:
			self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), \
				lr = self.run.config[LEARNING_RATE], weight_decay=self.cnn_params[L2_REGULATION_WEIGHT])
		elif self.cnn_params[OPTIMIZER] == OPTIMIZER_SGD:
			self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), \
				lr = self.cnn_params[LEARNING_RATE], momentum=0.9, weight_decay=self.cnn_params[L2_REGULATION_WEIGHT])
		self.scaler = GradScaler()
		if self.cnn_params[SHEDULER] == SHEDULER_PLATEAU :
			self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.2, patience=10, verbose=True)
		elif self.cnn_params[SHEDULER] == SHEDULER_STEP:
			self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.2)
		elif self.cnn_params[SHEDULER] == SHEDULER_COSINE:
			self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=40, eta_min=0)
			# TODO sheduler param 1 ,2 
		