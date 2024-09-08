import logging
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch

from beats_classifier import knn_model
from beats_classifier.beats_dataset import BEATsDataset
from MLHelper import constants as const
from MLHelper.dataset import AudioDataset
from MLHelper.ml_loop import HookManager, ML_Loop
from MLHelper.tools.utils import FileUtils, MLModelInfo, MLUtil
from run import Run


class BEATsTraining(ML_Loop):
	def __init__(self, run: Run, dataset: AudioDataset):
		super().__init__(run, dataset, BEATsDataset)
		self.run = run
		self.dataset = dataset
		self.model = None
		self.criterion = None
		self.beats_params = self.run.config[const.TRANSFORMER_PARAMS]
		self.knn_classifier = None
		self.is_knn_mode = self.beats_params[const.MODEL_SUB_TYPE] == const.MODEL_TYPE_KNN

	def set_training_utilities(self, start_model, optimizer, scheduler, scaler):
		self.model = start_model
		self.optimizer = optimizer
		self.scheduler = scheduler
		self.scaler = scaler

		if self.is_knn_mode:
			knn_params = self.run.config[const.KNN_PARAMS]
			self.knn_classifier = knn_model.KNN_Classifier(self.model, \
				knn_params, self.run.logger_dict[const.LOGGER_TENSOR], self.run.device)
			self.knn_classifier = self.knn_classifier.to(self.run.device)

	def start_training_task(self, start_epoch: int, start_fold: int):
		assert self.model is not None, "Model is None. Required for training"
		assert start_epoch >= 1, "Start epoch must be greater or equal to 0"
		if not self.is_knn_mode:
			assert self.optimizer is not None, "Optimizer is None. Required for training"
		return self.kfold_loop(start_epoch=start_epoch, start_fold=start_fold)

	@HookManager.hook_wrapper("training_step")
	def training_step(self, inputs: torch.Tensor, labels: torch.Tensor):
		if self.is_knn_mode:
			inputs = inputs.to(self.run.device)
			labels = labels.to(self.run.device)
			self.knn_classifier.add_training_example(inputs, labels)
			return None, None, labels
		return super().training_step(inputs, labels)

	@HookManager.hook_wrapper("validation_step")
	def validation_step(self, inputs: torch.Tensor, labels: torch.Tensor):
		if self.is_knn_mode:
			predictions, probabilities = self.knn_classifier(inputs)
			if isinstance(probabilities, np.ndarray):
				probabilities = torch.tensor(probabilities)
			return torch.tensor(77.7), probabilities, labels
		return super().validation_step(inputs, labels)
# TODO save knn model
	@HookManager.hook_wrapper("epoch")
	def epoch_loop(self, epoch: int, fold_idx: int, **kwargs: Any) -> None:
		if self.config[const.TASK_TYPE] == const.TRAINING:
			if self.is_knn_mode:
				self.training_epoch_loop(epoch=epoch, fold=fold_idx)
				self.knn_classifier.build_nn_classifier()
		self.validation_epoch_loop(epoch=epoch, fold=fold_idx)

	@HookManager.hook_wrapper("fold")
	def fold_loop(self, fold_idx: int, start_epoch: int, **kwargs: Any) -> None:
		if self.is_knn_mode:
			self.knn_classifier.reset_data()
		super().fold_loop(fold_idx, start_epoch, **kwargs)

	@HookManager.hook_wrapper("end_training_epoch")
	def end_training_epoch(self, epoch: int, fold: int, **kwargs: Any) -> None:
		if not self.is_knn_mode:
			super().end_training_epoch(epoch, fold, **kwargs)
		else:
			self.run.log(f"Finished collecting data for KNN in epoch {epoch}", \
				name=const.LOGGER_TRAINING, level=logging.INFO)

	@HookManager.hook_wrapper("end_validation_epoch")
	def end_validation_epoch(self, epoch: int, fold: int, **kwargs: Any) -> None:
		super().end_validation_epoch(epoch, fold, **kwargs)
		if self.is_knn_mode:
			self.run.log(f"KNN validation completed for epoch {epoch}", \
				name=const.LOGGER_TRAINING, level=logging.INFO)

	def save_model(self, epoch, fold):
		if self.is_knn_mode:
			model_name = const.get_model_filename(type=f"{const.BEATS}_knn", epoch=epoch, fold=fold)
			full_path = FileUtils.join([self.run.run_results_path, const.MODELS_FOLDER_NAME, model_name])
			FileUtils.safe_path_create(full_path)
			state_dict = self.knn_classifier.save_state_dict()
			torch.save(state_dict, full_path)
			self.run.log(f"Saved KNN model to {full_path}", name=const.LOGGER_TRAINING, level=logging.INFO)
		else:
			super().save_model(epoch, fold)
