import math

from torch import nn, zeros, no_grad

import MLHelper.constants as const
from MLHelper.tools.utils import MLUtil
from run import Run


def get_demo_input():
	return zeros(1, 1, 128, 128)

def get_model(run: Run):
	type = run.config[const.CNN_PARAMS][const.MODEL_SUB_TYPE]
	if type == 1:
		return CNN_Model_1(run)
	if type == 2:
		return CNN_Model_2(run)
	if type == 3:
		return CNN_Model_3(run)
	if type == 4:
		return CNN_Model_4(run)
	raise ValueError(f"Model type {type} not found")

class CNN_Base(nn.Module):
	def __init__(self, run: Run):
		super().__init__()
		run_config = run.config
		cnn_config = run_config[const.CNN_PARAMS]
		self.num_classes = run.task.dataset.num_classes
		self.target_samplerate = run.task.dataset.target_samplerate
		self.pDrop0 = cnn_config[const.DROP0]
		self.pDrop1 = cnn_config[const.DROP1]
		self.seconds = run_config[const.CHUNK_DURATION]
		self.n_mels = cnn_config[const.N_MELS]
		if cnn_config[const.ACTIVATION] == const.ACTIVATION_RELU:
			self.activation = nn.ReLU(inplace=True)
		elif cnn_config[const.ACTIVATION] == const.ACTIVATION_L_RELU:
			self.activation = nn.LeakyReLU(inplace=True)
		elif cnn_config[const.ACTIVATION] == const.ACTIVATION_SILU:
			self.activation = nn.SiLU(inplace=True)
		else:
			raise ValueError(f"Activation {cnn_config[const.ACTIVATION]} not supported")
		self.softmax = nn.Softmax(dim=1)
		self.tensor_logger = run.logger_dict[const.LOGGER_TENSOR]

	def initialize(self):
		MLUtil.reset_weights(self)
		self.apply(self.init_weights)

	def forward(self, x):
		self.tensor_logger.debug(f"Raw Forward Input shape: {x.shape}")
		self.batchsize = x.shape[0]
		if len(x.shape) == 3: # missing channel dimension
			x = x.unsqueeze(1)

		self.tensor_logger.info(f"Modified Forward INIT Input shape: {x.shape}")
		return x

	def kaiming_normal_silu(self, tensor, mode="fan_out"):
		fan = nn.init._calculate_correct_fan(tensor, mode)
		gain = 1.0  # Standardwert für lineare und die meisten anderen Aktivierungen
		gain *= math.sqrt(3)  # Anpassung für SiLU
		std = gain / math.sqrt(fan)
		with no_grad():
			return tensor.normal_(0, std)

	def init_weights(self, m):
		if isinstance(m, (nn.Conv2d, nn.Linear)):
			if isinstance(getattr(m, "activation", None), nn.SiLU):
				# Spezielle Initialisierung für SiLU
				self.kaiming_normal_silu(m.weight, mode="fan_out")
			else:
				# Standard Kaiming-Initialisierung für andere Aktivierungen (z.B. ReLU)
				nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

			if m.bias is not None:
				nn.init.constant_(m.bias, 0)
		elif isinstance(m, nn.BatchNorm2d):
			nn.init.constant_(m.weight, 1)
			nn.init.constant_(m.bias, 0)

class CNN_Model_1(CNN_Base):
	def __init__(self, run_config):
		super().__init__(run_config)
		conv_layers = []

		# First Convolutional Block
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3))
		self.bn1 = nn.BatchNorm2d(8)
		self.maxpool2d1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		conv_layers += [self.conv1, self.bn1, self.activation, self.maxpool2d1]

		# Second Convolutional Block
		self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3))
		self.bn2 = nn.BatchNorm2d(16)
		self.maxpool2d2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		conv_layers += [self.conv2, self.bn2, self.activation, self.maxpool2d2]

		# Third Convolutional Block
		self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3))
		self.bn3 = nn.BatchNorm2d(32)
		self.maxpool2d3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		conv_layers += [self.conv3, self.bn3, self.activation, self.maxpool2d3]

		# Fourth Convolutional Block
		self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
		self.bn4 = nn.BatchNorm2d(64)
		self.maxpool2d4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		conv_layers += [self.conv4, self.bn4, self.activation, self.maxpool2d4]

		self.ap = nn.AdaptiveAvgPool2d(output_size=(8,8))

		# Flatten the output of the convolutional layers
		self.flatten = nn.Flatten()

		# First Fully-Connected (Linear) Layer
		self.fc1 = nn.Linear(in_features=4096, out_features=1024)
		self.bn5 = nn.BatchNorm1d(1024)
		fc1_layers = [self.fc1, self.bn5, self.activation]
		if self.pDrop0 > 0.0:
			fc1_layers += [nn.Dropout(p=self.pDrop0)]

		self.fc2 = nn.Linear(in_features=1024, out_features=128)
		self.bn6 = nn.BatchNorm1d(128)
		fc2_layers = [self.fc2, self.bn6, self.activation]
		if self.pDrop1 > 0.0:
			fc2_layers += [nn.Dropout(p=self.pDrop1)]

		# Third Fully-Connected (Linear) Layer - Last layer with 2 outputs
		self.fc3 = nn.Linear(in_features=128, out_features=self.num_classes)

		self.conv = nn.Sequential(*conv_layers)
		self.fc1_block = nn.Sequential(*fc1_layers)
		self.fc2_block = nn.Sequential(*fc2_layers)

	def forward(self, x):
		x = super().forward(x)
		x = self.conv(x)
		x = self.ap(x)
		x = self.flatten(x)
		self.tensor_logger.debug(f"shape after flatten: {x.shape}")
		x = self.fc1_block(x)
		x = self.fc2_block(x)
		x = self.fc3(x)
		return x


# nur padding und kernel
class CNN_Model_2(CNN_Base):
	def __init__(self, run_config):
		super().__init__(run_config)
		conv_layers = []

		# First Convolutional Block
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(7, 7), padding=3)
		self.bn1 = nn.BatchNorm2d(8)
		self.maxpool2d1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		conv_layers += [self.conv1, self.bn1, self.activation, self.maxpool2d1]

		# Second Convolutional Block
		self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(5, 5), padding=2)
		self.bn2 = nn.BatchNorm2d(16)
		self.maxpool2d2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		conv_layers += [self.conv2, self.bn2, self.activation, self.maxpool2d2]

		# Third Convolutional Block
		self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3))
		self.bn3 = nn.BatchNorm2d(32)
		self.maxpool2d3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		conv_layers += [self.conv3, self.bn3, self.activation, self.maxpool2d3]

		# Fourth Convolutional Block
		self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))
		self.bn4 = nn.BatchNorm2d(64)
		self.maxpool2d4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		conv_layers += [self.conv4, self.bn4, self.activation, self.maxpool2d4]

		self.ap = nn.AdaptiveAvgPool2d(output_size=(8,8))

		# Flatten the output of the convolutional layers
		self.flatten = nn.Flatten()

		# First Fully-Connected (Linear) Layer
		self.fc1 = nn.Linear(in_features=4096, out_features=1024)
		self.bn5 = nn.BatchNorm1d(1024)
		fc1_layers = [self.fc1, self.bn5, self.activation]
		if self.pDrop0 > 0.0:
			fc1_layers += [nn.Dropout(p=self.pDrop0)]

		self.fc2 = nn.Linear(in_features=1024, out_features=128)
		self.bn6 = nn.BatchNorm1d(128)
		fc2_layers = [self.fc2, self.bn6, self.activation]
		if self.pDrop1 > 0.0:
			fc2_layers += [nn.Dropout(p=self.pDrop1)]

		# Third Fully-Connected (Linear) Layer - Last layer with 2 outputs
		self.fc3 = nn.Linear(in_features=128, out_features=self.num_classes)

		self.conv = nn.Sequential(*conv_layers)
		self.fc1_block = nn.Sequential(*fc1_layers)
		self.fc2_block = nn.Sequential(*fc2_layers)

	def forward(self, x):
		x = super().forward(x)
		x = self.conv(x)
		x = self.ap(x)
		x = self.flatten(x)
		self.tensor_logger.debug(f"shape after flatten: {x.shape}")
		x = self.fc1_block(x)
		x = self.fc2_block(x)
		x = self.fc3(x)
		return x


# mehr dimensionen
class CNN_Model_3(CNN_Base):
	def __init__(self, run_config):
		super().__init__(run_config)
		conv_layers = []

		# First Convolutional Block
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(7, 7), padding=3)
		self.bn1 = nn.BatchNorm2d(16)
		self.maxpool2d1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		conv_layers += [self.conv1, self.bn1, self.activation, self.maxpool2d1]

		# Second Convolutional Block
		self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), padding=2)
		self.bn2 = nn.BatchNorm2d(32)
		self.maxpool2d2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		conv_layers += [self.conv2, self.bn2, self.activation, self.maxpool2d2]

		# Third Convolutional Block
		self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
		self.bn3 = nn.BatchNorm2d(64)
		self.maxpool2d3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		conv_layers += [self.conv3, self.bn3, self.activation, self.maxpool2d3]

		# Fourth Convolutional Block
		self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1)
		self.bn4 = nn.BatchNorm2d(128)
		self.maxpool2d4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		conv_layers += [self.conv4, self.bn4, self.activation, self.maxpool2d4]

		self.conv = nn.Sequential(*conv_layers)

		self.ap = nn.AdaptiveAvgPool2d(output_size=(8,8))

		# Flatten the output of the convolutional layers
		self.flatten = nn.Flatten()

		# Berechne die Eingabegröße für die erste Fully Connected Layer
		dummy_input = zeros(1, 1, self.n_mels, int(self.target_samplerate * self.seconds))
		conv_out = self.conv(dummy_input)
		ap_out = self.ap(conv_out)
		fc_input_size = self.flatten(ap_out).shape[1]

		# Fully-Connected Layers
		self.fc1 = nn.Linear(in_features=fc_input_size, out_features=512)
		self.bn5 = nn.BatchNorm1d(512)
		fc1_layers = [self.fc1, self.bn5, self.activation, nn.Dropout(p=self.pDrop0)]

		self.fc2 = nn.Linear(in_features=512, out_features=128)
		self.bn6 = nn.BatchNorm1d(128)
		fc2_layers = [self.fc2, self.bn6, self.activation, nn.Dropout(p=self.pDrop1)]

		self.fc3 = nn.Linear(in_features=128, out_features=self.num_classes)

		self.fc1_block = nn.Sequential(*fc1_layers)
		self.fc2_block = nn.Sequential(*fc2_layers)

	def forward(self, x):
		x = super().forward(x)
		x = self.conv(x)
		x = self.ap(x)
		x = self.flatten(x)
		self.tensor_logger.debug(f"shape after flatten: {x.shape}")
		x = self.fc1_block(x)
		x = self.fc2_block(x)
		x = self.fc3(x)
		return x

# mehr FC
class CNN_Model_4(CNN_Base):
	def __init__(self, run_config):
		super().__init__(run_config)
		conv_layers = []

		# First Convolutional Block
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3))
		self.bn1 = nn.BatchNorm2d(8)
		self.maxpool2d1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		conv_layers += [self.conv1, self.bn1, self.activation, self.maxpool2d1]

		# Second Convolutional Block
		self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3))
		self.bn2 = nn.BatchNorm2d(16)
		self.maxpool2d2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		conv_layers += [self.conv2, self.bn2, self.activation, self.maxpool2d2]

		# Third Convolutional Block
		self.conv3 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(3, 3))
		self.bn3 = nn.BatchNorm2d(64)
		self.maxpool2d3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		conv_layers += [self.conv3, self.bn3, self.activation, self.maxpool2d3]

		# Fourth Convolutional Block
		self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3))
		self.bn4 = nn.BatchNorm2d(128)
		self.maxpool2d4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		conv_layers += [self.conv4, self.bn4, self.activation, self.maxpool2d4]

		self.conv = nn.Sequential(*conv_layers)

		self.ap = nn.AdaptiveAvgPool2d(output_size=(4,4))

		# Flatten the output of the convolutional layers
		self.flatten = nn.Flatten()

		# Berechne die Eingabegröße für die erste Fully Connected Layer
		dummy_input = zeros(1, 1, self.n_mels, int(self.target_samplerate * self.seconds))
		conv_out = self.conv(dummy_input)
		ap_out = self.ap(conv_out)
		fc_input_size = self.flatten(ap_out).shape[1]

		# First Fully-Connected (Linear) Layer
		self.fc1 = nn.Linear(in_features=fc_input_size, out_features=2048)
		self.bn5 = nn.BatchNorm1d(2048)
		fc1_layers = [self.fc1, self.bn5, self.activation]
		if self.pDrop0 > 0.0:
			fc1_layers += [nn.Dropout(p=self.pDrop0)]

		self.fc2 = nn.Linear(in_features=2048, out_features=512)
		self.bn6 = nn.BatchNorm1d(512)
		fc2_layers = [self.fc2, self.bn6, self.activation]
		if self.pDrop1 > 0.0:
			fc2_layers += [nn.Dropout(p=self.pDrop1)]

		# Third Fully-Connected (Linear) Layer - Last layer with 2 outputs
		self.fc3 = nn.Linear(in_features=512, out_features=self.num_classes)

		self.conv = nn.Sequential(*conv_layers)
		self.fc1_block = nn.Sequential(*fc1_layers)
		self.fc2_block = nn.Sequential(*fc2_layers)

	def forward(self, x):
		x = super().forward(x)
		x = self.conv(x)
		x = self.ap(x)
		x = self.flatten(x)
		self.tensor_logger.debug(f"shape after flatten: {x.shape}")
		x = self.fc1_block(x)
		x = self.fc2_block(x)
		x = self.fc3(x)
		return x
