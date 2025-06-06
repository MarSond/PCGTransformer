import logging

from torch.utils.data import DataLoader

import MLHelper.constants as const
from MLHelper.dataset import AudioDataset
from MLHelper.ml_loop import ML_Loop
from run import Run

from .cnn_dataset import CNN_Dataset


class CNNTraining(ML_Loop):

	def __init__(self, run: Run, dataset: AudioDataset) -> None:
		super().__init__(run, dataset, pytorch_dataset_class=CNN_Dataset)
		self.cnn_params = self.run.config[const.CNN_PARAMS]


	def num_worker_test(self, logger: logging.Logger):
		import multiprocessing as mp
		from time import time
		self.run_config = self.base_config
		train_list, valid_list = self._get_train_valid_list(None, None)
		dataset = CNN_Dataset(datalist=train_list, run=self.run)
		for num_workers in range(0, mp.cpu_count() + 1, 2):
			start = time()
			train_loader = DataLoader(
				dataset, shuffle=True, num_workers=num_workers, batch_size=64, drop_last=True)
			dl_it = iter(train_loader)
			for _ in range(1, 5):
				for _, (_data_batch, _labels) in enumerate(dl_it):
					pass
			end = time()
			logger.error(f"Finish with:{end - start} second, num_workers={num_workers}")

	def set_training_utilities(self, start_model, optimizer, scheduler, scaler):
		self.model = start_model
		self.optimizer = optimizer
		self.scheduler = scheduler
		self.scaler = scaler

	def start_training_task(self, start_epoch=1, start_fold=1):
		""" model is either a new model or a checkpointed model
		start_epoch is the epoch to start at.
		The loop beginns from start but skeips training untill start_epoch
		"""
		assert self.model is not None, "Model is None. Required for training"
		assert self.optimizer is not None, "Optimizer is None. Required for training"
		assert start_epoch >= 1, "Start epoch must be greater or equal to 0"
		duration_ms = \
			(self.config[const.CNN_PARAMS][const.N_FFT] / self.run.task.dataset.target_samplerate) * 1000
		self.run.log_training(f"FFT window duration: {duration_ms:.2f} ms", level=logging.INFO)
		return self.kfold_loop(start_epoch=start_epoch, start_fold=start_fold)
