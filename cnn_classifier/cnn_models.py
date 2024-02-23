import torch.nn as nn
import parameters as cfg
import logging

def get_model(run_config):
	type = run_config['model_type']
	if type == 1:
		return CNN_Model_1(run_config)
	elif type == 2:
		return CNN_Model_2(run_config)
	elif type == 3:
		return CNN_Model_3(run_config)
	else:
		raise ValueError(f"Model type {type} not found")

class CNN_Base(nn.Module):
	def __init__(self, run_config):
		super().__init__()
		self.num_classes = run_config['num_classes']
		self.target_samplerate = run_config['target_samplerate']
		self.pDrop0 = run_config['drop0']
		self.pDrop1 = run_config['drop1']
		self.seconds = run_config['seconds']
		self.n_mels = run_config['n_mels']
		if run_config['activation'] == "relu":
			self.activation = nn.ReLU(inplace=True)
		elif run_config['activation'] == "l_relu":
			self.activation = nn.LeakyReLU(inplace=True)	
		elif run_config['activation'] == "silu":
			self.activation = nn.SiLU(inplace=True)
		else: raise ValueError(f"Activation {run_config['activation']} not found in YAMNET Model list")
		self.softmax = nn.Softmax(dim=1)
		self.tensor_logger = logging.getLogger(cfg.TENSOR_LOGGER)
		self._initialize_weights()
	

	def forward(self, x):
		self.tensor_logger.debug(f"Raw Forward Input shape: {x.shape}")
		self.batchsize = x.shape[0]
		if len(x.shape) == 3: # missing channel dimension
			x = x.unsqueeze(1)
		self.tensor_logger.info(f"Modified Forward INIT Input shape: {x.shape}")
		return x

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='silu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
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
		x = self.fc1_block(x)	
		x = self.fc2_block(x)
		x = self.fc3(x)	
		return x
	
class CNN_Model_3(CNN_Base):
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
		self.fc1 = nn.Linear(in_features=4096, out_features=128)
		self.bn5 = nn.BatchNorm1d(128)
		fc1_layers = [self.fc1, self.bn5, self.activation]
		if self.pDrop0 > 0.0:
			fc1_layers += [nn.Dropout(p=self.pDrop0)]

		# Second Fully-Connected (Linear) Layer - Last layer with 2 outputs
		self.fc3 = nn.Linear(in_features=128, out_features=self.num_classes)

		self.conv = nn.Sequential(*conv_layers)
		self.fc1_block = nn.Sequential(*fc1_layers)

	def forward(self, x):
		x = super().forward(x)
		x = self.conv(x)
		x = self.ap(x)
		x = self.flatten(x)
		x = self.fc1_block(x)	
		x = self.fc3(x)	
		return x
	

class CNN_Model_2(CNN_Base):
	def __init__(self, run_config):
		super().__init__(run_config)
		
		self.drop0 = self.pDrop0
		self.drop1 = self.pDrop1
		
		# First Convolutional Block
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3))
		self.bn1 = nn.BatchNorm2d(8)
		self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		
		# Second Convolutional Block
		self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3))
		self.bn2 = nn.BatchNorm2d(16)
		self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		
		# Third Convolutional Block
		self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3))
		self.bn3 = nn.BatchNorm2d(32)
		self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
		
		# Adaptive Pooling
		self.ap = nn.AdaptiveAvgPool2d(output_size=(16, 16))  # Adjusted size
		
		# Flatten Layer
		self.flatten = nn.Flatten()
		
		# First Fully-Connected Layer
		self.fc1 = nn.Linear(in_features=8192, out_features=1024)  # Adjusted input size
		self.bn4 = nn.BatchNorm1d(1024)
		self.fc1_block = nn.Sequential(self.fc1, self.bn4, self.activation)
		if self.pDrop0 > 0:
			self.fc1_block.add_module('dropout0', nn.Dropout(p=self.pDrop0))
		
		# Second Fully-Connected Layer
		self.fc2 = nn.Linear(in_features=1024, out_features=128)
		self.bn5 = nn.BatchNorm1d(128)
		self.fc2_block = nn.Sequential(self.fc2, self.bn5, self.activation)
		if self.pDrop1 > 0:
			self.fc2_block.add_module('dropout1', nn.Dropout(p=self.pDrop1))
		
		# Output Layer
		self.fc3 = nn.Linear(in_features=128, out_features=self.num_classes)
		
	def forward(self, x):
		x = super().forward(x)
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.activation(x)
		x = self.maxpool1(x)
		
		x = self.conv2(x)
		x = self.bn2(x)
		x = self.activation(x)
		x = self.maxpool2(x)
		
		x = self.conv3(x)
		x = self.bn3(x)
		x = self.activation(x)
		x = self.maxpool3(x)
		
		x = self.ap(x)
		
		x = self.flatten(x)
		
		x = self.fc1_block(x)
		x = self.fc2_block(x)
		
		x = self.fc3(x)
		
		return x

