import torch

from data.dataset import AudioDataset
from MLHelper import constants as const
from MLHelper.ml_loop import ML_Loop, validation_epoch_hook
from run import Run

from .cnn_dataset import CNN_Dataset


class CNN_Inference(ML_Loop):

	def __init__(self, run: Run, dataset: AudioDataset) -> None:
		super().__init__(run, dataset, pytorch_dataset_class=CNN_Dataset)

	def prepare_kfold_run(self) -> None:
		pass

	def plot_batch(self, data, target):
		import matplotlib.pyplot as plt
		_, axs = plt.subplots(2, 5, figsize=(20, 10))
		for i in range(10):
			axs[i // 5, i % 5].imshow(data[i].squeeze().cpu().numpy(), cmap='gray')
			axs[i // 5, i % 5].set_title(f"Label: {target[i]}")
		plt.show()

	@validation_epoch_hook
	def validation_epoch_loop(self, epoch: int = 0, fold: int = 0) -> None:
		#self.prepare_fold(fold_idx)
		self.pbars.update_total(bar_name=self.pbars.NAME_VALID, total=len(self.valid_loader))

		y_true = []
		y_pred = []
		# self.metrics.reset_epoch_metrics(validation=True)
		for _, (data, target) in enumerate(self.valid_loader):
			# self.plot_batch(data, target)
			data, target = data.to(self.device), target.to(self.device)
			with torch.no_grad():
				loss, probabilities = self.predict_step( \
					model=self.model, inputs=data, labels=target)
				prediction = probabilities.argmax(dim=1)
				y_true += target.cpu().numpy().tolist()
				y_pred += prediction.cpu().numpy().tolist()
				self.metrics.update_step( \
					probabilities=probabilities, labels=target, loss=loss, validation=True)
			self.pbars.increment(bar_name=self.pbars.NAME_VALID)

			if self.run.config[const.SINGLE_BATCH_MODE]:
				break


	def start_inference_task(self, model: torch.nn.Module) -> None:
		self.model = model
		self.model.eval()

		self.kfold_loop(start_epoch=1)
