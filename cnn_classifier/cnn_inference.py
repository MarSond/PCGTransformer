import parameters as cfg
from .cnn_dataset import CNN_Dataset
import torch
from ml_helper import metrics
from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader
from os.path import join as pjoin
from ml_helper.metrics import MetricsTracker
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, matthews_corrcoef
from ml_helper import ml_base
from ml_helper.utils import MLLogging
from torch.cuda.amp import autocast

class CNN_Inference(ml_base.MLBase):

	def __init__(self, base_config, device, datalist):
		self.base_config = base_config
		self.device = device
		self.datalist = datalist
		self.batchsize = 1
		self.logger, self.tensor_logger = MLLogging.getLogger([cfg.MAIN_LOGGER, cfg.TENSOR_LOGGER])
		self.metrics = MetricsTracker(
			config=base_config, device=device, metrics_class=metrics.TorchMetricsAdapter)

	def get_model(self, path=None, name=None):
		model_base = self.base_config['model']
		if path is None:
			path = pjoin(model_base, self.base_config['model_path'])
		model = self.load_model(path=path, name=name)
		model.eval()
		model = model.to(self.device)
		self.model=model
		return model

	def get_dataloader(self):
		full_dataset = CNN_Dataset(self.datalist, run_config=self.base_config)
		full_dataset.set_mode("validation")
		testloader = DataLoader(full_dataset, batch_size=self.batchsize, shuffle=True, drop_last=True)
		return testloader

	# prediction of one batch
	def predict_step(self, inputs, labels):
		with autocast():	
			self.tensor_logger.info(f"predict_step inputs shape {inputs.shape}")
			outputs = self.model(inputs)

		probabilities = torch.softmax(outputs, dim=1)
		return probabilities


	def start_inference(self):
		model = self.model #get_model()
		self.model.eval()
		testloader = self.get_dataloader()
		pbar = tqdm(total=len(testloader), desc="Inference")
		y_true = []
		y_pred = []
		#self.metrics.reset_epoch_metrics(validation=True)
		for batch_idx, (data, target) in enumerate(testloader):
			data, target = data.to(self.device), target.to(self.device)
			with torch.no_grad():
				probabilities  = self.predict_step(data, target)
				prediction = probabilities.argmax(dim=1, keepdim=True)
				y_true += target.cpu().numpy().tolist()
				y_pred += prediction.cpu().numpy().tolist()
				#self.metrics.update_step(predictions=prediction, labels=y_pred, loss=None, validation=True)
			pbar.update(1)
			pbar.set_postfix_str(f"Batch: {batch_idx}/{len(testloader)} + Output: {prediction.cpu().numpy()} - Target: {target.cpu().numpy()}")
		pbar.close()
		#fold_metrics = self.metrics.save_epoch_metrics(validation=True)
		#self.metrics.finish_fold()
		#self.metrics.print_end_summary()
		accuracy = accuracy_score(y_true, y_pred)
		f1 = f1_score(y_true, y_pred, average='macro')
		precision = precision_score(y_true, y_pred, average='macro')
		recall = recall_score(y_true, y_pred, average='macro')
		specificity = recall_score(y_true, y_pred, pos_label=0, average='macro')
		confusion = confusion_matrix(y_true, y_pred)
		mcc = matthews_corrcoef(y_true, y_pred)
		
		print(f"Accuracy: {accuracy}")
		print(f"F1: {f1}")
		print(f"Precision: {precision}")
		print(f"Recall: {recall}")
		print(f"Specificity: {specificity}")
		print(f"Confusion Matrix: \n{confusion}")
		print(f"MCC: {mcc}")
		return accuracy, f1, precision, recall, specificity, confusion

