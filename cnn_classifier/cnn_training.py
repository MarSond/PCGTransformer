from .cnn_dataset import CNN_Dataset
import torch
from torch.utils.data import DataLoader
import pandas as pd
from MLHelper import constants as const
from data.dataset import AudioDataset
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

	def start_training_run(self):

		self.kfold_loop()
