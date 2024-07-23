import logging
from pathlib import Path

import torch
from torch import Tensor, nn

from beats_classifier.BEATs.BEATs import BEATs, BEATsConfig
from MLHelper import constants as const
from run import Run


class BEATsBase(nn.Module):
	def __init__(self, run: Run):
		super().__init__()
		self.run = run
		self.config = run.config
		self.beats_config = self.config[const.TRANSFORMER_PARAMS]
		self.num_classes = run.task.dataset.num_classes
		self.target_samplerate = run.task.dataset.target_samplerate
		self.tensor_logger = run.logger_dict[const.LOGGER_TENSOR]
		if self.beats_config[const.ACTIVATION] == const.ACTIVATION_RELU:
			self.activation = nn.ReLU(inplace=True)
		elif self.beats_config[const.ACTIVATION] == const.ACTIVATION_L_RELU:
			self.activation = nn.LeakyReLU(inplace=True)
		elif self.beats_config[const.ACTIVATION] == const.ACTIVATION_SILU:
			self.activation = nn.SiLU(inplace=True)
		else:
			raise ValueError(f"Activation {self.beats_config[const.ACTIVATION]} not supported")

		self.pDrop0 = self.beats_config[const.DROP0]
		self.pDrop1 = self.beats_config[const.DROP1]
		self.beats = self.setup_extractor()
		assert self.beats_config[const.FREEZE_EXTRACTOR] is True, "BEATs extractor must be frozen for now"

	def setup_extractor(self):
		path = Path(self.config[const.TRANSFORMER_PARAMS][const.EXTRACTOR_FOLDER]) / \
			self.config[const.TRANSFORMER_PARAMS][const.EXTRACTOR_NAME]
		self.run.log_training(f"Loading BEATs extractor from {path}", level=logging.INFO)
		checkpoint = torch.load(path)
		beats_config = BEATsConfig(checkpoint["cfg"])
		beats_model = BEATs(beats_config, logger=self.tensor_logger)
		beats_model.load_state_dict(checkpoint["model"])

		return beats_model

	def initialize(self):
		# Initialize BEATs model
		pass

	def forward(self, x: Tensor) -> Tensor:
		self.tensor_logger.debug(f"Raw Forward Input shape: {x.shape}")
		# TODO assert shapes

		return x

class BEATsModel(BEATsBase):
	def __init__(self, run: Run):
		super().__init__(run)

		# Replace the final layer for our classification task
		self.classifier = nn.Linear(self.beats.encoder.embedding_dim, self.num_classes)

		if self.config[const.TRANSFORMER_PARAMS][const.FREEZE_EXTRACTOR]:
			for param in self.beats.parameters():
				param.requires_grad = False
			self.beats.eval()

	def forward(self, x: Tensor, padding_mask=None) -> Tensor:
		x = super().forward(x)
		# Use BEATs extract_features method
		x, _ = self.beats.extract_features(x)
		self.tensor_logger.info(f"BEATs extract_features output shape: {x.shape}")
		# Mittelwertbildung über die Zeitdimension
		if padding_mask is not None and padding_mask.any():
			x = x.masked_fill(padding_mask.unsqueeze(-1), 0)
			x = x.sum(dim=1) / (~padding_mask).sum(dim=1).unsqueeze(-1)
		else:
			x = x.mean(dim=1)
		# Pass through the classifier
		self.tensor_logger.debug(f"BEATsModel classifier input shape: {x.shape}")

		x = self.classifier(x)

		self.tensor_logger.info(f"BEATsModel classifier output shape: {x.shape}")

		return x

def get_model(run: Run):
	model_sub_type = run.config[const.TRANSFORMER_PARAMS][const.MODEL_SUB_TYPE]
	if model_sub_type == 1:
		model = BEATsModel(run)
	elif model_sub_type == 2:
		model = BEATsModel2(run)
	else:
		raise ValueError(f"Model sub type {model_sub_type} not supported.")
	#model = torch.compile(model)

	return model

class BEATsModel2(BEATsBase):
	def __init__(self, run: Run):
		super().__init__(run)

		self.classifier = nn.Sequential(
			nn.Linear(self.beats.encoder.embedding_dim, 1024),
			nn.BatchNorm1d(1024),
			self.activation,
			nn.Dropout(self.pDrop0),
			nn.Linear(1024, 512),
			nn.BatchNorm1d(512),
            self.activation,
            nn.Dropout(self.pDrop1),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            self.activation,
            nn.Dropout(self.pDrop1),
            nn.Linear(256, self.num_classes)
        )

		if self.config[const.TRANSFORMER_PARAMS][const.FREEZE_EXTRACTOR]:
			for param in self.beats.parameters():
				param.requires_grad = False
			self.beats.eval()

	def forward(self, x: Tensor, padding_mask=None) -> Tensor:
		x = super().forward(x)
		x, _ = self.beats.extract_features(x)
		self.tensor_logger.info(f"BEATs extract_features output shape: {x.shape}")

		if padding_mask is not None and padding_mask.any():
			x = x.masked_fill(padding_mask.unsqueeze(-1), 0)
			x = x.sum(dim=1) / (~padding_mask).sum(dim=1).unsqueeze(-1)
		else:
			x = x.mean(dim=1)

		self.tensor_logger.debug(f"BEATsModel2 classifier input shape: {x.shape}")
		x = self.classifier(x)
		self.tensor_logger.info(f"BEATsModel2 classifier output shape: {x.shape}")

		return x

def get_demo_input():
	return torch.randn(1, 16000)
