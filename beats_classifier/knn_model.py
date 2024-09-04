import logging

import torch
from sklearn.neighbors import KNeighborsClassifier
from torch import nn
import numpy as np

from MLHelper import constants as const


class KNN_Classifier(nn.Module):

	def __init__(self, extractor, knn_params, logger=None, device="cpu"):
		super().__init__()
		self.tensor_logger = logger
		if self.tensor_logger is None:
			self.tensor_logger = logging.getLogger(__name__)
			self.tensor_logger.setLevel(logging.INFO)
			self.tensor_logger.warning("No logger provided. Using default logger.")
		self.extractor = extractor
		self.knn_params = knn_params
		self.device = device
		self.reset_data()
		self._set_params(knn_params)
		self.is_fitted = False
		self.classifier = None

	def _set_params(self, params: dict):
		self.n_neighbors = params[const.KNN_N_NEIGHBORS]
		self.knn_distance = params[const.KNN_METRIC]
		self.combine_mode = params[const.KNN_COMBINE_METHOD]
		self.weights = params[const.KNN_WEIGHT]
		self.assume_positive_p = params[const.KNN_ASSUME_POSTIVE_P]
		self.metric = params[const.KNN_METRIC]

	def build_nn_classifier(self):
		self.classifier = KNeighborsClassifier(n_neighbors=self.n_neighbors, weights=self.weights, metric=self.metric)
		assert self.n_neighbors is not None, "Number of neighbors not set."
		assert self.assume_positive_p is not None, "Assume positive probability not set."
		assert self.neighbor_data is not None, "No data to build nearest neighbor classifier with."

		neighbor_data_cpu = self.neighbor_data.cpu().detach().numpy()
		neighbor_labels_cpu = self.neighbor_labels.cpu().detach().numpy()

		self.tensor_logger.warning(	f"Building nearest neighbor classifier with {self.n_neighbors} neighbors, "
									f"distance metric {self.metric} and embedding mode {self.combine_mode} and "
									f"and assume_positive_p {self.assume_positive_p}")

		self.classifier.fit(neighbor_data_cpu, neighbor_labels_cpu)

		self.is_fitted = True

	def load_state_dict(self, state_dict):
		assert const.KNN_PARAMS in state_dict, "No knn params found in state dict."
		self.extractor = state_dict[const.EXTRACTOR]
		self.neighbor_data = state_dict[const.KNN_NEIGHBOR_DATA]
		self.neighbor_labels = state_dict[const.KNN_NEIGHBOR_LABELS]
		super().load_state_dict(state_dict[const.MODEL_STATE_DICT])
		self._set_params(state_dict[const.KNN_PARAMS])
		self.build_nn_classifier()  # Rebuild the nearest neighbor classifier

	def save_state_dict(self):
		state_dict = {
			const.EXTRACTOR: self.extractor,
			const.KNN_NEIGHBOR_DATA: self.neighbor_data,
			const.KNN_NEIGHBOR_LABELS: self.neighbor_labels,
			const.MODEL_STATE_DICT: super().state_dict(),
			const.KNN_PARAMS: self.knn_params
		}
		return state_dict

	def reset_data(self):
		self.neighbor_labels = torch.Tensor().to(self.device)
		self.neighbor_data = torch.Tensor().to(self.device)

	def add_neighbor_data(self, data, labels):
		self.neighbor_data = torch.cat([self.neighbor_data, data], dim=0)
		self.neighbor_labels = torch.cat([self.neighbor_labels, labels], dim=0)

	def extract_embeddings(self, x):
		if self.extractor is None:
			raise RuntimeError("The feature extractor is not set.")
		with torch.no_grad():
			embeddings = self.extractor(x)
		self.tensor_logger.info(f"raw embedding shape: {embeddings.shape}")
		embeddings = self.combine_embeddings(embeddings, self.combine_mode)
		self.tensor_logger.info(f"Features shape after combining: {embeddings.shape}")
		# bei beats kein combine nötig da es nur 1 embedding gibt
		batch_size, num_embeddings, embedding_size = embeddings.shape
		embeddings = embeddings.reshape(batch_size*num_embeddings, embedding_size)
		self.tensor_logger.info(f"Features shape after reshaping: {embeddings.shape}")
		return embeddings

	def add_training_example(self, inputs, labels):
		with torch.no_grad():
			embeddings = self.extract_embeddings(inputs)
		self.tensor_logger.info(f"embeddings device: {embeddings.device}")
		self.tensor_logger.info(f"inputs device: {inputs.device}")
		self.tensor_logger.info(f"self.neighbor_data device: {self.neighbor_data.device}")

		self.add_neighbor_data(embeddings, labels)

	@staticmethod
	def combine_embeddings(embeddings, embedding_mode):
		if embedding_mode == const.KNN_COMBINE_METHOD_MEAN: # TODO check die dimensionen und shapes
			embeddings = torch.mean(embeddings, dim=1)
		elif embedding_mode == const.KNN_COMBINE_METHOD_TRIPPLE_MEAN: # mean the 3 successive embeddings
			if embeddings.shape[1] % 3 != 0:
				remainder = embeddings.shape[1] % 3
				# mean the last embeddings
				last_embeddings = torch.mean(embeddings[:, -remainder:], dim=1, keepdim=True)
				# remove the last embeddings before the reshape
				embeddings = embeddings[:, :-remainder]
				# Reshape and mean in 3 stacked blocks
				embeddings = embeddings.reshape(embeddings.shape[0], -1, 3, embeddings.size()[2])
				embeddings = torch.mean(embeddings, dim=2)
				embeddings = torch.cat([embeddings, last_embeddings], dim=1)
			else:
				embeddings = embeddings.reshape(embeddings.shape[0], -1, 3, embeddings.size()[2])
				embeddings = torch.mean(embeddings, dim=2)
		elif embedding_mode == const.KNN_COMBINE_METHOD_MAX:
			embeddings = torch.max(embeddings, dim=1)[0]
		elif embedding_mode == const.KNN_COMBINE_METHOD_MIN:
			embeddings = torch.min(embeddings, dim=1)[0]
		elif embedding_mode == const.KNN_COMBINE_METHOD_SUM:
			embeddings = torch.sum(embeddings, dim=1)
		elif embedding_mode == const.KNN_COMBINE_METHOD_PLAIN:
			pass
		elif embedding_mode == const.KNN_COMBINE_METHOD_MEDIAN:
			embeddings = torch.median(embeddings, dim=1)[0]
		else:
			raise ValueError(f"Invalid embedding mode: {embedding_mode}")
		if len(embeddings.shape) == 2:
			embeddings = embeddings[:, None, :]
		return embeddings

	def forward(self, x):
		assert isinstance(x, torch.Tensor), "Input must be a tensor."
		self.tensor_logger.debug(f"Raw predict_proba input shape: {x.shape}")
		if not self.is_fitted:
			raise RuntimeError("The KNN classifier has not been fitted yet.")

		embeddings = self.extract_embeddings(x) # already combined
		num_embeddings = 1
		batch_size, embedding_size = embeddings.shape
		self.tensor_logger.debug(f"batch_size: {batch_size}, num_embeddings: {num_embeddings}, embedding_size: {embedding_size}")
		embeddings = embeddings.reshape(batch_size*num_embeddings, embedding_size)
		embeddings = embeddings.cpu().detach().numpy()
		self.tensor_logger.debug(f"Embeddings shape after forward reshape: {embeddings.shape}")

		# Klassifikation und Wahrscheinlichkeiten
		preds = self.classifier.predict(embeddings)
		preds_proba = self.classifier.predict_proba(embeddings)
		preds_reshape = preds.reshape(batch_size, num_embeddings)
		preds_proba_reshape = preds_proba.reshape(batch_size, num_embeddings, -1)  # Anzahl der Klassen
		self.tensor_logger.info(f"Predictions shape: {preds_reshape.shape}")
		self.tensor_logger.info(f"Predictions proba shape: {preds_proba_reshape.shape}")
		# Schwellenwert-Logik
		if self.combine_mode in [const.KNN_COMBINE_METHOD_PLAIN, const.KNN_COMBINE_METHOD_TRIPPLE_MEAN]:
			positive_counts = np.sum(preds_reshape, axis=1)
			threshold = num_embeddings * self.assume_positive_p
			instance_predictions = np.where(positive_counts >= threshold, 1, 0)
		else:
			instance_predictions = np.argmax(np.mean(preds_proba_reshape, axis=1), axis=-1)  # Multiclass

		instance_probabilities = np.mean(preds_proba_reshape, axis=1)

		return instance_predictions, instance_probabilities
