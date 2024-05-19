import parameters as cfg
from .cnn_dataset import CNN_Dataset
from . import cnn_models
from ml_helper.utils import MLUtil, MLPbar
from torch.utils.data import DataLoader
from cnn_base import CNN_base
import logging

class CNN_Training():

	def __init__(self, base_config, device, datalist, pbars: MLPbar = None):
		super().__init__(base_config, device, datalist, pbars)

	def get_model(self):
		return cnn_models.get_model(self.base_config)

	
	
	def num_worker_test(self, logger:logging.Logger):
		from time import time
		import multiprocessing as mp
		self.run_config = self.base_config
		train_list, valid_list = self._get_train_valid_list(None, None)
		dataset = CNN_Dataset(datalist=train_list, run=self.run)
		for num_workers in range(0, mp.cpu_count()+1, 2):  
			start = time()
			train_loader = DataLoader(dataset, shuffle=True, num_workers=num_workers, batch_size=64, drop_last=True)
			dl_it = iter(train_loader)
			for epoch in range(1, 5):
				for _, (data_batch, labels) in enumerate(dl_it):
					pass
			end = time()
			logger.error("Finish with:{} second, num_workers={}".format(end - start, num_workers))
