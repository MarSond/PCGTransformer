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


class DataBundle:
	def __init__(self, x, y=None, is_training=True):
		self.X = x
		self.y = y
		self.is_training = is_training

class XyPipeline(Pipeline):
	def fit(self, x, y=None):
		data = DataBundle(x, y, is_training=True)
		for _, transform in self.steps[:-1]:
			data = transform.fit_transform(data)
		self.steps[-1][1].fit(data.X, data.y)
		return self

	def transform(self, x, is_training=False):
		data = DataBundle(x, is_training=is_training)
		for _, transform in self.steps[:-1]:
			data = transform.transform(data)
		return data.X

	def predict(self, x):
		x_transformed = self.transform(x, is_training=False)
		return self.steps[-1][1].predict(x_transformed)

	def predict_proba(self, x):
		x_transformed = self.transform(x, is_training=False)
		return self.steps[-1][1].predict_proba(x_transformed)

class XyTransformerMixin:
	def fit_transform(self, data: DataBundle) -> DataBundle:
		self.fit(data)
		return self.transform(data)

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

	def build_pipeline(self):
		steps = []

		if self.emb_params.get(const.USE_HDBSCAN, False):
			assert const.HDBSCAN_PARAM_MIN_CLUSTER_SIZE in self.emb_params, "HDBSCAN min_cluster_size not set."
			assert const.HDBSCAN_PARAM_MIN_SAMPLES in self.emb_params, "HDBSCAN min_samples not set."
			steps.append(("hdbscan", HDBSCANTransformer(
				min_cluster_size=self.emb_params.get(const.HDBSCAN_PARAM_MIN_CLUSTER_SIZE),
				min_samples=self.emb_params.get(const.HDBSCAN_PARAM_MIN_SAMPLES)
			)))

		if self.emb_params[const.USE_SMOTE]:
			steps.append(("smote", SMOTETransformer(random_state=const.SEED_VALUE)))

		if self.emb_params.get(const.REDUCE_DIM_UMAP, False):
			assert const.EMBEDDINGS_REDUCE_UMAP_N_COMPONENTS in self.emb_params, "UMAP n_components not set."
			assert const.EMBEDDINGS_REDUCE_UMAP_N_NEIGHBORS in self.emb_params, "UMAP n_neighbors not set."
			assert const.EMBEDDINGS_REDUCE_UMAP_MIN_DIST in self.emb_params, "UMAP min_dist not set."
			steps.append(("umap", UMAPTransformer(
				n_components=self.emb_params.get(const.EMBEDDINGS_REDUCE_UMAP_N_COMPONENTS),
				n_neighbors=self.emb_params.get(const.EMBEDDINGS_REDUCE_UMAP_N_NEIGHBORS),
				min_dist=self.emb_params.get(const.EMBEDDINGS_REDUCE_UMAP_MIN_DIST)
			)))

		# select the classifier (kNN, Random Forest, etc.)
		if self.emb_params[const.EMBEDDING_CLASSIFIER] == const.CLASSIFIER_KNN:
			assert const.KNN_N_NEIGHBORS in self.emb_params, "kNN n_neighbors not set."
			assert const.KNN_WEIGHT in self.emb_params, "kNN weight not set."
			assert const.KNN_METRIC in self.emb_params, "kNN metric not set."
			steps.append(("classifier", KNNClassifierWrapper(
				n_neighbors=self.emb_params[const.KNN_N_NEIGHBORS],
				weights=self.emb_params[const.KNN_WEIGHT],
				metric=self.emb_params[const.KNN_METRIC],
				n_jobs=-1
			)))
		else:
			raise NotImplementedError( \
				f"Embedding classifier {self.emb_params[const.EMBEDDING_CLASSIFIER]} not implemented.")

		self.pipeline = XyPipeline(steps)

	def fit_pipeline(self):
		"""
		Called in training loop after last batch was fed into and pipeline was created.
		"""
		assert self.pipeline is not None, "Pipeline not set."
		assert self.embedding_data is not None, "No data to build nearest neighbor classifier with."
		if self.embedding_data.ndim == 1:
			self.embedding_data = self.embedding_data.reshape(-1, 1)

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
			const.EMBEDDING_PARAMS: self.emb_params,
			const.SEED: const.SEED_VALUE
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


		# Use transform method directly for inference (which includes validation)
		transformed_embeddings = self.pipeline.transform(combined_embeddings, is_training=False)

		predictions = self.pipeline.steps[-1][1].predict(transformed_embeddings)
		probabilities = self.pipeline.steps[-1][1].predict_proba(transformed_embeddings)

		return predictions, probabilities


class UMAPTransformer(BaseEstimator, XyTransformerMixin):
	def __init__(self, n_components=2, n_neighbors=15, min_dist=0.1):
		self.n_components = n_components
		self.n_neighbors = n_neighbors
		self.min_dist = min_dist
		self.umap_model = None

	def fit(self, data: DataBundle):
		self.umap_model = umap.UMAP(
			n_components=self.n_components,
			n_neighbors=self.n_neighbors,
			min_dist=self.min_dist,
			#random_state=const.SEED_VALUE,
			low_memory=False,
			n_jobs=-1,
		)
		self.umap_model.fit(data.X)
		return self

	def transform(self, data: DataBundle):
		x_transformed = self.umap_model.transform(data.X)
		return DataBundle(x_transformed, data.y)

class HDBSCANTransformer(BaseEstimator, XyTransformerMixin):
	def __init__(self, min_cluster_size=5, min_samples=5):
		self.min_cluster_size = min_cluster_size
		self.min_samples = min_samples
		self.clusterer = None
		self.outlier_mask = None

	def fit(self, data: DataBundle):
		self.clusterer = hdbscan.HDBSCAN(
			min_cluster_size=self.min_cluster_size,
			min_samples=self.min_samples,
		)
		cluster_labels = self.clusterer.fit_predict(data.X)
		self.outlier_mask = cluster_labels != -1
		return self

	def transform(self, data: DataBundle):
		if not data.is_training:
			return data  # Skip HDBSCAN during validation
		x_filtered = data.X[self.outlier_mask]
		y_filtered = data.y[self.outlier_mask] if data.y is not None else None
		return DataBundle(x_filtered, y_filtered)

class SMOTETransformer(BaseEstimator, XyTransformerMixin):
	def __init__(self, random_state=None):
		self.random_state = random_state
		self.smote = SMOTE(random_state=self.random_state)

	def fit(self, data: DataBundle):
		return self

	def transform(self, data: DataBundle):
		if not data.is_training:
			return data  # Skip SMOTE during validation
		if data.y is not None:
			x_resampled, y_resampled = self.smote.fit_resample(data.X, data.y)
			return DataBundle(x_resampled, y_resampled)
		return data

class KNNClassifierWrapper(BaseEstimator, TransformerMixin):
	def __init__(self, n_neighbors=5, weights="uniform", metric="minkowski", algorithm="auto", n_jobs=-1):
		self.n_neighbors = n_neighbors
		self.weights = weights
		self.metric = metric
		self.algorithm = algorithm
		self.n_jobs = n_jobs
		self.classifier = KNeighborsClassifier(
			n_neighbors=self.n_neighbors,
			weights=self.weights,
			metric=self.metric,
			algorithm=self.algorithm,
			n_jobs=self.n_jobs,
		)

	def fit(self, x, y):
		self.classifier.fit(x, y)
		return self

	def transform(self, x):
		# Return the distance to each neighbor for each point
		return self.classifier.kneighbors(x)[0]

	def predict(self, x):
		return self.classifier.predict(x)

	def predict_proba(self, x):
		return self.classifier.predict_proba(x)
