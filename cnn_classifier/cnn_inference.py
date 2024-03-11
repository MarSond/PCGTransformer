from .cnn_dataset import CNN_Dataset
import torch
from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, matthews_corrcoef
import pandas as pd


class CNN_Inference():

	def __init__(self, run, model, datalist: pd.DataFrame):
		self.run = run
		self.model = model
		self.base_config = run.config
		self.device = run.device
		self.datalist = datalist
		self.batchsize = 1
		self.dataset_base = run.task.dataset.dataset_path

	def get_dataloader(self):
		full_dataset = CNN_Dataset(datalist=self.datalist, run=self.run)
		full_dataset.set_mode("validation")
		testloader = DataLoader(full_dataset, batch_size=self.batchsize, shuffle=True, drop_last=False)
		return testloader

	# prediction of one batch
	def start_inference(self):
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

