from .cnn_dataset import CNN_Dataset
import torch
from MLHelper.ml_loop import *
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, matthews_corrcoef
import pandas as pd
from MLHelper import constants as const
from data.dataset import AudioDataset
from run import Run
from MLHelper.ml_loop import ML_Loop


class CNN_Inference(ML_Loop):

	def __init__(self, run: Run, dataset: AudioDataset) -> None:
		super().__init__(run, dataset, pytorch_dataset_class=CNN_Dataset)
		# _, self.valid_loader = self.dataset.get_dataloaders(num_split=1, Torch_Dataset_Class=CNN_Dataset)

	def prepare_self(self) -> None:
		return super().prepare_self()

	def plot_batch(self, data, target):
		import matplotlib.pyplot as plt
		import numpy as np
		fig, axs = plt.subplots(2, 5, figsize=(20, 10))
		for i in range(10):
			axs[i // 5, i % 5].imshow(data[i].squeeze().cpu().numpy(), cmap='gray')
			axs[i // 5, i % 5].set_title(f"Label: {target[i]}")
		plt.show()

	@fold_hook
	def fold_loop(self, fold_idx: int, **kwargs: Any) -> None:
		#self.prepare_fold(fold_idx)
		self.pbars.update_total(bar_name=self.pbars.NAME_VALID, total=len(self.valid_loader))

		y_true = []
		y_pred = []
		# self.metrics.reset_epoch_metrics(validation=True)
		for batch_idx, (data, target) in enumerate(self.valid_loader):
			# self.plot_batch(data, target)
			data, target = data.to(self.device), target.to(self.device)
			with torch.no_grad():
				loss, probabilities = self.predict_step(model=self.model, inputs=data, labels=target)
				prediction = probabilities.argmax(dim=1, keepdim=True)
				y_true += target.cpu().numpy().tolist()
				y_pred += prediction.cpu().numpy().tolist()
				# self.metrics.update_step(predictions=prediction, labels=y_pred, loss=None, validation=True)
			self.pbars.increment(bar_name=self.pbars.NAME_VALID)

			if self.run.config[const.SINGLE_BATCH_MODE]:
				break
		# fold_metrics = self.metrics.save_epoch_metrics(validation=True)
		# self.metrics.finish_fold()
		# self.metrics.print_end_summary()
		accuracy = accuracy_score(y_true, y_pred)
		f1 = f1_score(y_true, y_pred, average='macro')
		precision = precision_score(y_true, y_pred, average='macro')
		recall = recall_score(y_true, y_pred, average='macro')
		tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
		specificity = tn / (tn + fp)
		confusion = confusion_matrix(y_true, y_pred)
		mcc = matthews_corrcoef(y_true, y_pred)
		nmcc = (mcc + 1) / 2
		pred_count_per_class = pd.Series(y_pred).value_counts()

		# TODO PLR NLR
		print(f"Accuracy: {accuracy}")
		print(f"F1: {f1}")
		print(f"Precision: {precision}")
		print(f"Recall: {recall}")
		print(f"Specificity: {specificity}")
		print(f"Confusion Matrix: \n{confusion}")
		print(f"\033[1m\033[91m Norm-MCC: {nmcc} \033[0m")
		print(f"Predictions per class:\n{pred_count_per_class}")
		print(f"True labels per class:\n{pd.Series(y_true).value_counts()}")
		print(const.CLASS_DESCRIPTION)
		return accuracy, f1, precision, recall, specificity, confusion, nmcc

	# prediction of one batch

	def start_inference_task(self, model):
		self.model = model
		self.model.eval()

		self.kfold_loop(start_epoch=1)
