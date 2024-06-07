# helper functions like "predict", the k fold loop and other things. Creates dataloaders etc.
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from cnn_dataset import CNN_Dataset
from run import Run
from MLHelper.constants import *
from MLHelper.ml_loop import ML_Loop
from data.dataset import AudioDataset


class CNN_Loop(ML_Loop):

	def __init__(self, run: Run, dataset: AudioDataset) -> None:
		super().__init__(run=run, dataset=dataset)
	
	

	