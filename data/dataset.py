import os
from os.path import join as pjoin
from os.path import normpath
import pandas as pd

class Dataset:

	def __init__(self):
		self.base_path = os.path.dirname(__file__)
		pass

class Physionet2016(Dataset):

	def __init__(self):
		super().__init__()
		self.folder_name = 'physionet2016'
		self.dataset_path = pjoin(self.base_path, self.folder_name)
		self.meta_file_train = pjoin(self.dataset_path, 'train_list.csv')
		self.meta_file_test = pjoin(self.dataset_path, 'test_list.csv')
		self.train_audio_path = f"{self.dataset_path}/audiofiles/train/*/*.wav"
		self.num_classes = 2

	

class Physionet2022(Dataset):

	def __init__(self):
		super().__init__()
		self.folder_name = 'physionet2022'
		self.meta_file_train = 'train_list.csv'
		self.meta_file_test = 'test_list.csv'
		self.dataset_path = pjoin(self.base_path, self.folder_name)
		self.train_audio_path = f"{self.dataset_path}/audiofiles/train/*/*.wav"
		self.num_classes = 2

def get_file_metadata_human(path: str, anno: pd.DataFrame, data_classes: pd.DataFrame, dataset_object: Physionet2016):
	meta = { }
	id = Path(path).name.split(".")[0]
	dataset_name = str(Path(path).parents[0]).split("\\")[-1]
	info = torchaudio.info(path)
	audio = torchaudio.load(path)
	
	try: quality = data_classes.loc[id]['quality'].squeeze()
	except KeyError:
		quality = 1
	if np.isnan(quality):
		quality = 1	# Assume quality is okay if no data is given
	
	######### Get diagnosis #########
	try: diagnosis = anno.loc[id]['Diagnosis']
	except KeyError:
		# diagnosis not found - likely id not in annotation list - use old name from other list
		new_id = anno['Original record name'] == id
		val = anno.loc[new_id]["Diagnosis"].squeeze()
		if len(val) == 0:  
			val = "Unknown"
		diagnosis = str(val)

	meta["path"] = path.replace(dataset_object.dataset_path + "\\","")
	
	#meta["path"] = path.replace(paths['train_audio_path'].replace("/","\\") + "\\","")
	meta["name"] = id
	meta["dataset"] = dataset_name
	meta['diagnosis'] = diagnosis	# NOT the label, but the diagnosis string
	meta["quality"] = int(quality)
	meta["sr"] = info.sample_rate 
	meta["channels"] = info.num_channels 
	meta["length"] = info.num_frames
	meta["bits"] = info.bits_per_sample 
	#meta["min_amp"] = audio[0].min().item()
	#meta["max_amp"] = audio[0].max().item()
	try:    # Try "updated" annotation list
		dataclass = int(anno.loc[id]['Class (-1=normal 1=abnormal)'].squeeze())
	except KeyError:    # If fails, use original dataset annotation
		dataclass = int(data_classes.loc[id]['class'].squeeze())
	if dataclass == -1:
		dataclass = 0
	meta["label_1"] = dataclass
	return meta

def parse_physionet2016():
	"""
	Parse the folder structure of the Physionet 2016 dataset and create meta files
	"""
	dataset = Physionet2016()
	train_data = glob.glob(dataset.train_audio_path)
	train_data = [normpath(path) for path in train_data]
	dataset_names = ['training-a', 'training-b', 'training-c', 'training-d', 'training-e', 'training-f']
	name_list = pd.DataFrame()
	
	for name in dataset_names:
		subset_liste = glob.glob(f"{dataset.dataset_path}/audiofiles/train/{name}/*.wav")
		print(f"{name} count: {len(subset_liste)}")
		for path_name in subset_liste:
			name = Path(path_name).stem
			#concat name to name_list using concat function
			name_list = pd.concat([name_list, pd.DataFrame([name], columns=["name"])], ignore_index=True)
			
	name_list.set_index("name", inplace=True)
	print("Total count: ", len(name_list))

	anno = pd.read_csv(pjoin(dataset.dataset_path, "Online_Appendix_training_set.csv"))
	anno.set_index("Challenge record name", inplace=True)

	print("All annotations count", len(anno))

	data_classes = pd.read_csv(pjoin(dataset.dataset_path, "classes_SQI.csv"))
	data_classes.set_index("name",inplace=True)
	print(data_classes.head())
	annotation_classes = anno[["Class (-1=normal 1=abnormal)"]]
	print(annotation_classes.head())
	error = []
	for row in name_list.iterrows():
		name = row[0]
		if name in annotation_classes.index:
			anno_class = annotation_classes.loc[name].values[0]
			data_c = data_classes.loc[name].get("class")
			if anno_class == data_c:
				pass
				#print(f"Name {name} found in annotations and classes and is equal. {anno_class} == {data_c}")
			else:
				print(f"Name {name} found in annotations and classes but is not equal. {anno_class} != {data_c}")
		else:
			print(f"Name {name} not found in regular annotations")
			if name in anno.get("Original record name").values:
				orig_class = anno.loc[anno["Original record name"] == name].get("Class (-1=normal 1=abnormal)").values[0]
				data_c = data_classes.loc[name].get("class")
				print(f"BUT Name {name} found in Original record name with class {orig_class} and is in data_classes with class {data_c}")
			else:
				print(f"Name {name} WAS NOT Original names")
				if name in data_classes.index:
					print(f"Name {name} found in data_classes with class {data_classes.loc[name].get('class')}")
					error.append(name)
				else:
					print(f"Name {name} WAS NOT in ANYTHING")
					
	print("Len error: ", len(error))


	print("All train data count", len(train_data))
	print("All annotations count", len(anno))
	print("All classes count", len(data_classes))
	metadata = []

	pbar = tqdm.tqdm(total=len(train_data), position=0, leave=True, desc="Human data list")
	for file in train_data:
		metadata.append(get_file_metadata_human(file, anno, data_classes, dataset))
		pbar.update(1)

		

	metadata_df = pd.DataFrame(metadata)
	print(metadata_df.head())
	print("Train path", dataset.meta_file_train)
	metadata_df.to_csv(dataset.meta_file_train, index=True, index_label="id", encoding='utf-8')
	


if __name__ == '__main__':
	# parse folder structure and create meta files
	from os.path import normpath
	import numpy as np
	import glob
	from pathlib import Path  
	import pandas as pd
	import tqdm.autonotebook as tqdm
	import torchaudio
	parse_physionet2016()
