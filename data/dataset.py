import os
from os.path import join as pjoin
from os.path import normpath
import pandas as pd

class Dataset:

	def __init__(self):
		self.base_path = os.path.dirname(__file__)
		self.columns = ["label_1", "patient_id", "path", "dataset", "diagnosis", "quality", "sr", "channels", "length", "bits"]
		pass

	def load_dataset(self) -> pd.DataFrame:
		dataframe = pd.read_csv(self.meta_file_train, index_col="id", encoding='utf-8')
		self.datalist = dataframe
		return dataframe
	
	# check if get attribute datalist is loaded
	def __getattr__(self, name):
		if name == "datalist":
			return self.load_dataset()
		else:
			raise AttributeError(f"Attribute {name} not found")

class Physionet2016(Dataset):

	def __init__(self):
		super().__init__()
		self.folder_name = 'physionet2016'
		self.dataset_path = pjoin(self.base_path, self.folder_name)
		self.meta_file_train = pjoin(self.dataset_path, 'train_list.csv')
		self.meta_file_test = pjoin(self.dataset_path, 'test_list.csv')

		self.train_audio_base_folder = f"{self.dataset_path}/audiofiles/train/"
		self.train_audio_search_pattern = f"{self.dataset_path}/audiofiles/train/*/*.wav"
		self.num_classes = 2
		self.target_samplerate = 2000

	

class Physionet2022(Dataset):

	def __init__(self):
		super().__init__()
		self.folder_name = 'physionet2022'
		self.dataset_path = pjoin(self.base_path, self.folder_name)
		self.meta_file_train = pjoin(self.dataset_path, 'train_list.csv')
		self.meta_file_test = pjoin(self.dataset_path, 'test_list.csv')
		self.train_audio_base_folder = f"{self.dataset_path}/training_data/"
		self.train_audio_search_pattern = f"{self.dataset_path}/training_data/*.wav"
		self.num_classes = 2
		self.target_samplerate = 4000

def save_training_data_to_csv(metadata, dataset):
	metadata_df = pd.DataFrame(metadata)
	print(metadata_df.head())
	print("Train path", dataset.meta_file_train)
	metadata_df.to_csv(dataset.meta_file_train, index=True, index_label="id", encoding='utf-8')

def get_file_metadata_human_ph2016(path: str, anno: pd.DataFrame, data_classes: pd.DataFrame, dataset_object: Physionet2016) -> dict:
	meta = { key: None for key in dataset_object.columns }
	id = Path(path).name.split(".")[0]
	dataset_name = str(Path(path).parents[0]).split("\\")[-1]
	info = torchaudio.info(path)
	
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
	meta["patient_id"] = id
	meta["dataset"] = dataset_object.folder_name+"-"+dataset_name
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
	train_data = glob.glob(dataset.train_audio_search_pattern)
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
					raise ValueError(f"Name {name} WAS NOT in ANYTHING")
					
	print("Len error: ", len(error))


	print("All train data count", len(train_data))
	print("All annotations count", len(anno))
	print("All classes count", len(data_classes))
	metadata = []

	pbar = tqdm.tqdm(total=len(train_data), position=0, leave=True, desc="Human data list")
	for file in train_data:
		metadata.append(get_file_metadata_human_ph2016(file, anno, data_classes, dataset))
		pbar.update(1)

	save_training_data_to_csv(metadata, dataset)

def get_file_metadata_human_ph2022(path: str, anno: pd.DataFrame, dataset_object: Physionet2022) -> dict:
	meta = { key: None for key in dataset_object.columns }
	info = torchaudio.info(path)
	meta["path"] = path.replace(dataset_object.dataset_path + "\\","")
	patient_id = Path(path).name.split(".")[0].split("_")[0] # first block till _ from file name
	patient_id = int(patient_id)
	#if patient_id in anno.get("Additional ID").values:
	#	patient_id = patient_id + "+" + anno[anno.get("Additional ID") == patient_id].index[0]
	# TODO beachten dass additional ID nur einseitig beschrieben wird -> map oder so als vergleich, beide Fälle müssen gleiche ID haben
	meta["patient_id"] = patient_id
	meta["dataset"] = dataset_object.folder_name
	#meta['diagnosis'] = diagnosis	# not given
	#meta["quality"] = int(quality)	# not given
	meta["sr"] = info.sample_rate 
	meta["channels"] = info.num_channels 
	meta["length"] = info.num_frames
	meta["bits"] = info.bits_per_sample 
	#meta["min_amp"] = audio[0].min().item()
	#meta["max_amp"] = audio[0].max().item()
	
	#dataclass = anno.loc[id]['Murmur'].squeeze()
	# find the class in the annotation, by collumn "Patient ID"
	dataclass = anno.get("Murmur").loc[patient_id]

	if dataclass == "Absent":
		dataclass = 0
	elif dataclass == "Present":
		dataclass = 1
	elif dataclass == "Unknown":
		dataclass = 2
	else:
		raise ValueError(f"Unknown class {dataclass}")
	meta["label_1"] = dataclass
	return meta
# TODO addidional ID beachten

def parse_physionet2022():
	#https://moody-challenge.physionet.org/2022/#data-table
	print("Start parsing Physionet 2022")
	dataset = Physionet2022()
	annotation = pd.read_csv(pjoin(dataset.dataset_path, "training_data.csv"), index_col="Patient ID")
	print(annotation.head())
	print("Annotation types", annotation.dtypes)
	print("Looking for train data in", dataset.train_audio_search_pattern)
	annotation.index = annotation.index.astype(int)
	training_files = glob.glob(dataset.train_audio_search_pattern)
	training_files = [normpath(path) for path in training_files]
	print("All existing audio files count", len(training_files))
	print("All annotations count", len(annotation))
	metadata = []
	pbar = tqdm.tqdm(total=len(training_files), position=0, leave=True, desc="Physionet 2022 Human data list")
	for file in training_files:
		meta = get_file_metadata_human_ph2022(file, annotation, dataset)
		pbar.update(1)
		metadata.append(meta)
		
	metadata_df = pd.DataFrame(metadata)
	print(metadata_df.head())
	print("Train path", dataset.meta_file_train)
	metadata_df.to_csv(dataset.meta_file_train, index=True, index_label="id", encoding='utf-8')

def dataset_statistics(dataset: Dataset):
	train_data = dataset.load_dataset()
	print("Train data count", len(train_data))
	print("Classes count", train_data["label_1"].value_counts())
	print("Dataset count", train_data["dataset"].value_counts())
	print("Quality count", train_data["quality"].value_counts())
	print("SR count", train_data["sr"].value_counts())
	print("Channels count", train_data["channels"].value_counts())
	print("Length count", train_data["length"].value_counts())
	print("Bits count", train_data["bits"].value_counts())

def plot_statistics(dataset: Dataset):
	import matplotlib.pyplot as plt
	import seaborn as sns
	from mpl_toolkits.axes_grid1 import ImageGrid
	import io
	data = dataset.load_dataset()
	p_size=12
	font_size=26
	def make_plots(plot_data, dataset: Dataset, name=""):
		
		fig, axs = plt.subplots(1, 3, figsize=(p_size*3, p_size))
		seconds = plot_data["length"] / dataset.target_sample_rate
		CLASSES_1 = {0: "NORMAL", 1: "ABNORMAL", 2: "UNKNOWN"}
		CLASS_LABEL_1 = "label_1" 		# abnormal / normal
		CLASS_COLORS_1 = ["green", "red", "blue"]
		# Length plot with class colors
		bar_colors = [CLASS_COLORS_1[label] for label in plot_data[CLASS_LABEL_1].values]
		axs[0].bar(range(len(seconds)), seconds, color=bar_colors, width=1.0)
		axs[0].set_title(f"Length {name}", fontsize=font_size)
		axs[0].set_xlabel("Sample", fontsize=font_size)
		axs[0].set_ylabel("Seconds", fontsize=font_size)
		# increase tick size
		axs[0].tick_params(axis='both', which='major', labelsize=font_size-4)
		
		
		# Histogram plot using sns.histplot
		sns.histplot(data=seconds, bins=200, kde=True, ax=axs[1], color="blue")
		axs[1].set_title(f"Length histogram {name}", fontsize=font_size)
		axs[1].set_xlabel("Seconds", fontsize=font_size)
		axs[1].set_ylabel("Count", fontsize=font_size)
		axs[1].tick_params(axis='both', which='major', labelsize=font_size-4)

		#############
		# Create a sub-figure for pie chart and statistics
		subfig, subaxs = plt.subplots(2, 1, figsize=(p_size, p_size))
		# margin and padding 0
		subfig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

		
		class_counts = plot_data[CLASS_LABEL_1].value_counts(sort=False)
		# better class_counts which guarantees the order of the classes 0 then 1
		class_counts = [class_counts[0], class_counts[1]]
		subaxs[0].pie(class_counts, labels=[CLASSES_1[0], CLASSES_1[1]], autopct='%1.1f%%', colors=CLASS_COLORS_1, textprops={'fontsize': font_size-1})
		subaxs[0].set_title(f"Class balance {name}", fontsize=font_size+3)
		subaxs[0].set_aspect('equal')  # Make the pie chart circular
		# make sub plot left centered
		
		# Statistical information
		negatives = seconds[plot_data[CLASS_LABEL_1] == 0]
		positives = seconds[plot_data[CLASS_LABEL_1] == 1]
		mean_duration_neg = np.mean(negatives)
		mean_duration_pos = np.mean(positives)
		total_count = len(seconds)

		text_content = f"Average length (Negative): {mean_duration_neg:.2f}s\n"
		text_content += f"Average length (Positive): {mean_duration_pos:.2f}s\n"
		text_content += f"Average length (All): {np.mean(seconds):.2f}s\n"
		text_content += f"Standard deviation (Negative): {np.std(seconds[plot_data[CLASS_LABEL_1] == 0]):.2f}s\n"
		text_content += f"Standard deviation (Positive): {np.std(seconds[plot_data[CLASS_LABEL_1] == 1]):.2f}s\n"
		text_content += f"Standard deviation (All): {np.std(seconds):.2f}s\n"
		text_content += f"Median length: {np.median(seconds):.2f}s\n"
		text_content += f"Minimum length: {np.min(seconds):.2f}s\n"
		text_content += f"Maximum length: {np.max(seconds):.2f}s\n"
		text_content += f"Correlation class<>length: {plot_data[CLASS_LABEL_1].corr(seconds):.4f}\n"
		text_content += f"Counts: neg.: {len(negatives)} pos.: {len(positives)} total: {total_count}/3240"
		
		subaxs[1].text(-0.00, 0.5, text_content, fontsize=font_size+9, ha="left", va="center")
		subaxs[1].axis('off')

		# Adjust spacing in the sub-figure
		subfig.subplots_adjust(hspace=0.0, wspace=0.0, left=0.0, right=1.0, bottom=0.0, top=1.0)
		
		# Add the sub-figure to the main figure
		buf = io.BytesIO()
		subfig.savefig(buf, format='png')
		buf.seek(0)
		img = plt.imread(buf)
		axs[2].imshow(img)
		axs[2].axis('off')
		#############
		return fig


	figs = []
	dataset_names = data.dataset.unique()
	for name in dataset_names:
		data_dataset = data[data.dataset == name]
		print(f"Dataset {name}", f"{len(data_dataset)}/{len(data)}")
		# plot length + pie chart class balance + plot histogram of length
		figs.append(make_plots(data_dataset, name=name, dataset=dataset))

	# Combine all figs into a single image grid (2, 3)
	fig = plt.figure(figsize=(p_size*3*3, p_size*2))
	grid = ImageGrid(fig, 111, nrows_ncols=(3, 2), axes_pad=0.5)

	for i, (ax, im) in enumerate(zip(grid, figs)):
		ax.cla()
		ax.axis('off')
		
		buf = io.BytesIO()
		im.savefig(buf, format='png')
		buf.seek(0)
		
		img = plt.imread(buf)
		ax.imshow(img)

		ax.set_title(f"Dataset {dataset_names[i]}", fontsize=font_size+3)

	plt.tight_layout()
	plt.show()

	fig = make_plots(data, name="All datasets", dataset=dataset)
	plt.show()


if __name__ == '__main__':
	# parse folder structure and create meta files
	from os.path import normpath
	import numpy as np
	import glob
	from pathlib import Path  
	import pandas as pd
	import tqdm.autonotebook as tqdm
	import torchaudio
	parse_physionet2022()
	parse_physionet2016()
	print("Done parsing")
	#dataset_statistics(Physionet2016())
	dataset_statistics(Physionet2022())
	#plot_statistics(Physionet2022())
