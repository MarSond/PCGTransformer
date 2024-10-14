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

class BEATsTrainingKNN(ML_Loop):


	def __init__(self, run: Run, dataset: AudioDataset):
		super().__init__(run, dataset, BEATsDataset)
		self.run = run
		self.dataset = dataset
		self.model = None
		self.beats_params = self.run.config[const.TRANSFORMER_PARAMS]
		self.emb_classifier = None
		self.load_embeddings_run_name = self.run.config.get(const.LOAD_EMBEDDINGS_FROM_RUN_NAME)

	def set_training_utilities(self, start_model, optimizer, scheduler, scaler):
		self.model = start_model
		# self.optimizer = optimizer
		# self.scheduler = scheduler
		# self.scaler = scaler
		knn_params = self.run.config[const.EMBEDDING_PARAMS]
		self.emb_classifier = embedding_model.EmbeddingClassifier(self.model, \
			knn_params, self.run.logger_dict[const.LOGGER_TENSOR], self.run.device)
		self.emb_classifier = self.emb_classifier.to(self.run.device)

	def start_training_task(self, start_epoch: int, start_fold: int):
		assert self.model is not None, "Model is None. Required for training"
		assert start_epoch >= 1, "Start epoch must be greater or equal to 0"
		return self.kfold_loop(start_epoch=start_epoch, start_fold=start_fold)

	@HookManager.hook_wrapper("training_step")
	def training_step(self, inputs: torch.Tensor, labels: torch.Tensor):
		assert self.load_embeddings_run_name is None, "Load embeddings directly, not via training loop"
		inputs = inputs.to(self.run.device)
		labels = labels.to(self.run.device)
		self.emb_classifier.add_training_example(inputs, labels)
		return None, None, labels

	@HookManager.hook_wrapper("validation_step")
	def validation_step(self, inputs: torch.Tensor, labels: torch.Tensor):
		predictions, probabilities = self.emb_classifier(inputs)
		if isinstance(probabilities, np.ndarray):
			probabilities = torch.tensor(probabilities)
		return torch.tensor(999999.99), probabilities, labels

	def load_embeddings_from_file(self, fold):
		if self.load_embeddings_run_name:
			base_path = Path(self.run.config[const.RUN_FOLDER]) / self.load_embeddings_run_name / const.OTHER_FOLDER_NAME
			pkl_path = base_path / f"{const.FOLD}{fold}_{const.FILENAME_EMBEDDINGS_VALUE}"

			if pkl_path.exists():
				with pkl_path.open("rb") as file:
					data = pickle.load(file)
					self.emb_classifier.embedding_data = data[const.EMBEDDINGS]
					self.emb_classifier.embedding_labels = data[const.LABELS]
					assert data[const.FOLD] == fold
				self.run.log_training(f"Embeddings loaded from {pkl_path}", level=logging.INFO)
			else:
				self.run.log_training(f"Embeddings file not found: {pkl_path}", level=logging.ERROR)
				raise FileNotFoundError(f"Embeddings file not found: {pkl_path}")

	@HookManager.hook_wrapper("epoch_loop")
	def epoch_loop(self, epoch: int, fold_idx: int, **kwargs: Any) -> None:
		if self.config[const.TASK_TYPE] == const.TRAINING:
				if self.load_embeddings_run_name is None:
					self.training_epoch_loop(epoch=epoch, fold=fold_idx)
				else:
					self.load_embeddings_from_file(fold=fold_idx)
				self.emb_classifier.build_pipeline()
				self.emb_classifier.fit_pipeline()
				self.create_umap_plots_hook(epoch=epoch, fold=fold_idx)
		self.validation_epoch_loop(epoch=epoch, fold=fold_idx)

	def fold_loop(self, fold_idx: int, start_epoch: int, **kwargs: Any) -> None:
		self.emb_classifier.reset_data()
		super().fold_loop(fold_idx, start_epoch, **kwargs)

	def save_embeddings_to_file(self, embeddings, labels, fold):
		if self.config[const.EMBEDDING_PARAMS].get(const.EMBEDDING_SAVE_TO_FILE, False):
			base_path = Path(self.run.run_results_path) / const.OTHER_FOLDER_NAME
			pkl_path = base_path / f"{const.FOLD}{fold}_{const.FILENAME_EMBEDDINGS_VALUE}"
			FileUtils.safe_path_create(pkl_path)
			with (pkl_path).open("wb") as file:
				pickle.dump({const.EMBEDDINGS: embeddings, const.LABELS: labels, const.FOLD: fold}, file)
			self.run.log_training(f"Embeddings saved to {pkl_path}", level=logging.INFO)
		else:
			self.run.log_training(f"Embeddings not saved", level=logging.INFO)

	def create_umap_plots(self, embeddings, labels, epoch, fold, n_neighbors=30, min_dist=0.4, name="all_data"):
		import umap

		if embeddings.ndim == 1:
			embeddings = embeddings.reshape(-1, 1)
		reducer = umap.UMAP(random_state=const.SEED_VALUE, n_neighbors=n_neighbors, n_components=2, min_dist=min_dist, low_memory=False)
		umap_embeddings = reducer.fit_transform(embeddings) # no y=labels here, unsupervised UMAP
		umap_embeddings_supervised = reducer.fit_transform(embeddings, y=labels)

		fig_2d = Plotting.DimensionReduction.plot_umap2d(umap_embeddings, labels, n_neighbors=n_neighbors, min_dist=min_dist)
		fig_2d_supervised = Plotting.DimensionReduction.plot_umap2d(umap_embeddings_supervised, labels, n_neighbors=n_neighbors, min_dist=min_dist)

		base_path = Path(self.run.run_results_path) / const.OTHER_FOLDER_NAME

		FileUtils.safe_path_create(base_path)

		Plotting.show_save(fig_2d, base_path / f"umap_2d_{name}_epoch_{epoch}_fold_{fold}.png", show=False)
		Plotting.show_save(fig_2d_supervised, base_path / f"umap_2d_supervised_{name}_epoch_{epoch}_fold_{fold}.png", show=False)

		self.run.log_training(f"UMAP plots saved for epoch {epoch}, fold {fold}", level=logging.INFO)

# TODO text: methoden wie smote wurden bei CNN statdessen mittels FocalLoss angegangen
	def create_umap_plots_hook(self, epoch: int, fold: int, **kwargs: Any) -> None:
		self.run.logger_dict[const.LOGGER_TRAINING].info("Creating UMAP plots")
		embeddings = self.emb_classifier.embedding_data
		labels = self.emb_classifier.embedding_labels
		if not self.load_embeddings_run_name:
			self.save_embeddings_to_file(embeddings, labels, fold)
			self.create_umap_plots(embeddings, labels, epoch, fold)
		try:
			classifier = self.emb_classifier.pipeline.named_steps["classifier"].classifier
			self.create_umap_plots(classifier._fit_X, classifier._y, epoch, fold, name="after_transformation")
		except Exception as e:
			self.run.logger_dict[const.LOGGER_TRAINING].warning(f"Failed to create UMAP plots: {e}")

	def save_model(self, epoch, fold):
		if not self.config[const.SAVE_MODEL]:
			return
		model_name = const.get_model_filename(type=f"{const.BEATS}_knn", epoch=epoch, fold=fold)
		full_path = FileUtils.join([self.run.run_results_path, const.MODELS_FOLDER_NAME, model_name])
		FileUtils.safe_path_create(full_path)
		state_dict = self.emb_classifier.save_state_dict()
		torch.save(state_dict, full_path)
		self.run.log(f"Saved KNN model to {full_path}", name=const.LOGGER_TRAINING, level=logging.INFO)
