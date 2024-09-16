import logging
from typing import Optional

import hdbscan
import numpy as np
import torch
import umap
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KNeighborsClassifier
from torch import nn

from MLHelper import constants as const


class EmbeddingClassifier(nn.Module):

	def __init__(self, extractor: nn.Module, emb_params: dict, logger: Optional[logging.Logger], device="cpu"):
		super().__init__()
		self.tensor_logger = logger
		if self.tensor_logger is None:
			self.tensor_logger = logging.getLogger(__name__)
			self.tensor_logger.setLevel(logging.INFO)
			self.tensor_logger.warning("No logger provided. Using default logger.")
		# extractor, whos forward function generates batchsize x num_batches x emb_size embeddings
		self.extractor: nn.Module = extractor

		self.emb_params = emb_params
		self.device = device
		self.reset_data()

	def _create_nn_classifier(self) -> KNeighborsClassifier:
		assert self.emb_params[const.KNN_N_NEIGHBORS] is not None, "Number of neighbors not set."
		assert self.emb_params[const.KNN_WEIGHT] is not None, "Weight not set."
		assert self.emb_params[const.KNN_METRIC] is not None, "KNN Metric not set."

		classifier = KNeighborsClassifier(
			n_neighbors=self.emb_params[const.KNN_N_NEIGHBORS],
			weights=self.emb_params[const.KNN_WEIGHT],
			metric=self.emb_params[const.KNN_METRIC],
			algorithm=self.emb_params[const.KNN_ALGORITHM],
			n_jobs=-1
		)
		self.tensor_logger.warning( \
			f"Building nearest neighbor classifier with {self.emb_params[const.KNN_N_NEIGHBORS]} neighbors")
		return classifier

	def build_pipeline(self):
		steps = []

		if self.emb_params.get(const.REDUCE_DIM_UMAP, False):
			assert const.EMBEDDINGS_REDUCE_UMAP_N_COMPONENTS in self.emb_params, "UMAP n_components not set."
			assert const.EMBEDDINGS_REDUCE_UMAP_N_NEIGHBORS in self.emb_params, "UMAP n_neighbors not set."
			assert const.EMBEDDINGS_REDUCE_UMAP_MIN_DIST in self.emb_params, "UMAP min_dist not set."

			steps.append(("umap", UMAPTransformer(
				n_components=self.emb_params.get(const.EMBEDDINGS_REDUCE_UMAP_N_COMPONENTS, 2),
				n_neighbors=self.emb_params.get(const.EMBEDDINGS_REDUCE_UMAP_N_NEIGHBORS, 15),
				min_dist=self.emb_params.get(const.EMBEDDINGS_REDUCE_UMAP_MIN_DIST, 0.1)
			)))

		if self.emb_params.get(const.USE_HDBSCAN, False):
			steps.append(("hdbscan", HDBSCANTransformer(
				min_cluster_size=self.emb_params.get(const.HDBSCAN_PARAM_MIN_CLUSTER_SIZE, 5),
				min_samples=self.emb_params.get(const.HDBSCAN_PARAM_MIN_SAMPLES, 5)
			)))

		if self.emb_params[const.USE_SMOTE]:
			steps.append(("smote", SMOTE(random_state=const.SEED_VALUE)))

		# select the classifier (kNN, Random Forest, etc.)
		if self.emb_params[const.EMBEDDING_CLASSIFIER] == const.CLASSIFIER_KNN:
			steps.append(("classifier", self._create_nn_classifier()))
		else:
			raise NotImplementedError( \
				f"Embedding classifier {self.emb_params[const.EMBEDDING_CLASSIFIER]} not implemented.")

		self.pipeline = Pipeline(steps)

	def fit_pipeline(self):
		"""
		Called in training loop after last batch was fed into and pipeline was created.
		"""
		assert self.pipeline is not None, "Pipeline not set."
		assert self.embedding_data is not None, "No data to build nearest neighbor classifier with."
		if self.embedding_data.ndim == 1:
			self.embedding_data = self.embedding_data.reshape(-1, 1)

		# Sicherstellen, dass embedding_labels ein 2D-Array ist
		#if self.embedding_labels.ndim == 1:
		#	self.embedding_labels = self.embedding_labels.reshape(-1, 1)
		self.tensor_logger.info(f"Initial embedding_data shape: {self.embedding_data.shape}")
		self.tensor_logger.info(f"Initial embedding_labels shape: {self.embedding_labels.shape}")
		self.pipeline.fit(self.embedding_data, self.embedding_labels)
		self.tensor_logger.info(f"Final embedding_data shape: {self.embedding_data.shape}")
		self.tensor_logger.info(f"Final embedding_labels shape: {self.embedding_labels.shape}")

		self.tensor_logger.error("Pipeline fitted.")
		self.is_fitted = True

	def load_state_dict(self, state_dict):
		assert const.EMBEDDING_PARAMS in state_dict, "No knn params found in state dict."
		self.extractor = state_dict[const.EXTRACTOR]
		self.embedding_data = state_dict[const.EMBEDDING_DATA]
		self.embedding_labels = state_dict[const.EMBEDDING_LABELS]
		super().load_state_dict(state_dict[const.MODEL_STATE_DICT])
		self.emb_params = torch.load(state_dict[const.EMBEDDING_PARAMS])
		self.build_pipeline()
		self.fit_pipeline()

	def save_state_dict(self) -> dict:
		state_dict = {
			const.EXTRACTOR: self.extractor.state_dict(),
			const.EMBEDDING_DATA: self.embedding_data,
			const.EMBEDDING_LABELS: self.embedding_labels,
			const.MODEL_STATE_DICT: super().state_dict(),
			const.EMBEDDING_PARAMS: self.emb_params
		}
		return state_dict

	def reset_data(self):
		"""
		Called after new fold is created.
		"""
		self.embedding_data = np.empty((0, 1))  # Initialisiere als leeres 2D-Array
		self.embedding_labels = np.empty((0,))
		self.pipeline = None
		self.is_fitted = False

	def append_neighbor_data(self, data, labels):
		if isinstance(data, torch.Tensor):
			data = data.detach().cpu().numpy()
		if isinstance(labels, torch.Tensor):
			labels = labels.detach().cpu().numpy()

		# Ensure data is 2D
		if data.ndim == 1:
			data = data.reshape(1, -1)

		# Ensure labels is 1D
		labels = labels.ravel()

		# Debug info
		self.tensor_logger.debug(f"Appending data shape: {data.shape}, labels shape: {labels.shape}")
		self.tensor_logger.debug(f"Current embedding_data shape: {self.embedding_data.shape}")
		self.tensor_logger.debug(f"Current embedding_labels shape: {self.embedding_labels.shape}")

		# Append data
		if self.embedding_data.size == 0:
			self.embedding_data = data
		else:
			self.embedding_data = np.vstack((self.embedding_data, data))

		self.embedding_labels = np.concatenate((self.embedding_labels, labels))

		# Verify shapes
		assert self.embedding_data.shape[0] == self.embedding_labels.shape[0], "Mismatch in samples count"
		assert self.embedding_data.shape[1] == 768, f"Unexpected embedding size: {self.embedding_data.shape[1]}"

	def forward_extractor(self, input) -> np.ndarray:
		if self.extractor is None:
			raise RuntimeError("The feature extractor is not set.")
		with torch.no_grad():
			embeddings: torch.Tensor = self.extractor(input)
		self.tensor_logger.info(f"raw embedding shape from extractor: {embeddings.shape}")
		# immediate detach and convert to numpy and only use cpu afterwards
		embeddings: np.ndarray = embeddings.detach().cpu().numpy()
		if embeddings.ndim == 2:
			embeddings = embeddings[:, np.newaxis, :]
		return embeddings

	def extract_and_combine_embeddings(self, input: torch.Tensor) -> np.ndarray:
		embeddings: np.ndarray = self.forward_extractor(input)
		embeddings: np.ndarray = \
			self.combine_embeddings(embeddings, self.emb_params[const.EMBEDDING_COMBINE_METHOD])
		self.tensor_logger.info(f"Features shape after combining: {embeddings.shape}")
		return embeddings

	def add_training_example(self, inputs: torch.Tensor, labels: torch.Tensor):
		assert isinstance(inputs, torch.Tensor), "Input must be a tensor for add_training_example."
		"""
		called from training_step
		"""
		embeddings: np.ndarray = self.extract_and_combine_embeddings(inputs)
		self.append_neighbor_data(embeddings, labels)

	@staticmethod
	def combine_embeddings(embeddings: np.ndarray, embedding_mode: str) -> np.ndarray:
		assert isinstance(embeddings, np.ndarray), "Embeddings must be a numpy array in combine_embeddings."
		if embeddings.ndim == 2:
			embeddings = embeddings[:, np.newaxis, :]

		batch_size, num_embeddings, embedding_dim = embeddings.shape
		if embedding_mode == const.EMBEDDING_COMBINE_METHOD_MEAN:
			embeddings = np.mean(embeddings, axis=1)
		elif embedding_mode == const.EMBEDDING_COMBINE_METHOD_MAX:
			embeddings = np.max(embeddings, axis=1)
		elif embedding_mode == const.EMBEDDING_COMBINE_METHOD_MIN:
			embeddings = np.min(embeddings, axis=1)
		elif embedding_mode == const.EMBEDDING_COMBINE_METHOD_SUM:
			embeddings = np.sum(embeddings, axis=1)
		elif embedding_mode == const.EMBEDDING_COMBINE_METHOD_MEDIAN:
			embeddings = np.median(embeddings, axis=1)
		else:
			raise ValueError(f"Invalid embedding mode: {embedding_mode}")
		return embeddings

	def forward(self, x: torch.Tensor):
		assert isinstance(x, torch.Tensor), "Input must be a tensor."
		self.tensor_logger.debug(f"Raw predict_proba input shape: {x.shape}")
		if not self.is_fitted:
			raise RuntimeError("The KNN classifier has not been fitted yet.")

		embeddings: np.ndarray = self.forward_extractor(x)
		self.tensor_logger.info(f"Embeddings shape after extraction: {embeddings.shape}")

		combined_embeddings = self.combine_embeddings(embeddings, self.emb_params[const.EMBEDDING_COMBINE_METHOD])
		self.tensor_logger.warning(f"Embeddings shape after combining: {embeddings.shape}")


		predictions = self.pipeline.predict(combined_embeddings)
		probabilities = self.pipeline.predict_proba(combined_embeddings)

		return predictions, probabilities

class UMAPTransformer(BaseEstimator, TransformerMixin):

	def __init__(self, n_components=2, n_neighbors=15, min_dist=0.1):
		self.n_components = n_components
		self.n_neighbors = n_neighbors
		self.min_dist = min_dist
		self.umap_model = None

	def fit(self, x, y=None):
		self.umap_model = umap.UMAP(
			n_components=self.n_components,
			n_neighbors=self.n_neighbors,
			min_dist=self.min_dist,
			#random_state=const.SEED_VALUE,
			low_memory=False,
			n_jobs=-1,
		)
		self.umap_model.fit(x)
		return self

	def transform(self, x):
		output = self.umap_model.transform(x)
		return output

class HDBSCANTransformer(BaseEstimator, TransformerMixin):

	def __init__(self, min_cluster_size=5, min_samples=5):
		self.min_cluster_size = min_cluster_size
		self.min_samples = min_samples
		self.clusterer = None

	def fit(self, x, y=None):
		self.clusterer = hdbscan.HDBSCAN(
			min_cluster_size=self.min_cluster_size,
			min_samples=self.min_samples,
		)
		self.clusterer.fit(x)
		return self

	def transform(self, x, y=None):
		if self.clusterer is None:
			raise ValueError("Clusterer is not fitted yet.")
		cluster_labels = self.clusterer.fit_predict(x)
		mask = cluster_labels != -1
		if y is not None:
			return x[mask], y[mask]
		return x[mask]

	def fit_transform(self, x, y=None):
		self.fit(x)
		transformed = self.transform(x, y)
		return transformed
