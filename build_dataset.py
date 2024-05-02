from data.dataset import AudioDataset, Physionet2016, Physionet2022
import pandas as pd
import numpy as np
import torchaudio
from pathlib import Path
from os.path import normpath, join as pjoin
import tqdm.autonotebook as tqdm
import glob
import MLHelper.constants as const
from run import Run
import matplotlib.pyplot as plt
import json

def save_training_data_to_csv(metadata, dataset):
	metadata_df = pd.DataFrame(metadata)
	print(metadata_df.head())
	print("Train path", dataset.meta_file_train)
	metadata_df.to_csv(dataset.meta_file_train, index=True, index_label="id", encoding='utf-8')


def get_base_metadata(path):
	meta = { key: None for key in AudioDataset.columns }
	info = torchaudio.info(path)
	meta[const.META_SAMPLERATE] = info.sample_rate 
	meta[const.META_CHANNELS] = info.num_channels 
	meta[const.META_LENGTH] = info.num_frames
	meta[const.META_BITS] = info.bits_per_sample 
	return meta


def get_file_metadata_human_ph2016(path: str, anno: pd.DataFrame, data_classes: pd.DataFrame, dataset_object: Physionet2016) -> dict:
	meta = get_base_metadata(path)

	id = Path(path).name.split(".")[0]
	dataset_name = str(Path(path).parents[0]).split("\\")[-1]
	
	
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
	#meta["min_amp"] = audio[0].min().item()
	#meta["max_amp"] = audio[0].max().item()
	try:    # Try "updated" annotation list
		dataclass = int(anno.loc[id]['Class (-1=normal 1=abnormal)'].squeeze())
	except KeyError:	# If fails, use original dataset annotation
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
	metadata[const.META_HEARTCYCLES] = metadata[const.META_HEARTCYCLES].apply(json.dumps)
	save_training_data_to_csv(metadata, dataset)


def _get_heartcycle_indicies_2022(file_path: str) -> list:
	# something like "physionet2022/training_data/100_1.wav"
	base_path = Path(file_path).parents[0]
	audio_filename = Path(file_path).name.removesuffix(".wav")
	heartcycle_file = pjoin(base_path, f"{audio_filename}.tsv") 
	assert Path(heartcycle_file).exists(), f"File {heartcycle_file} does not exist"
	''' Example file content:
		0	0.242	0
		0.242	0.400176	1
		0.400176	0.497088	2
		0.497088	0.620176	3
		0.620176	0.780176	4
		0.780176	0.920176	1
		0.920176	1.060176	2
		1.060176	1.200176	3
		1.200176	1.340176	4
	1.340176	1.460176	1
	1.460176	1.600176	2 
	
	First column is start, second is end, third is the type
	0 means unclear. 1 means S1 wave, which is what we will use
	'''
	s1_start_times = []
	s1_end_times = []

	with open(heartcycle_file, "r") as f:
		lines = f.readlines()

	# Durchgehen jeder Zeile und Erfassen der Start- und Endzeiten von S1
	for line in lines:
		parts = line.strip().split("\t")
		if len(parts) < 3:
			raise ValueError(f"Line {line} does not have 3 parts")
		
		wave_type = parts[2]
		if wave_type == '1':  # Überprüfung, ob der Wellentyp '1' für S1 ist
			start_time = float(parts[0])
			end_time = float(parts[1])
			s1_start_times.append(start_time)
			s1_end_times.append(end_time)

	# Hinzufügen des Endwerts des letzten S1-Zyklus, falls vorhanden
 	# Nein , da Ende vom S1 zyklus nicht der ende vom gesamten Zyklus ist
	#if s1_end_times:
	#	s1_start_times.append(s1_end_times[-1])

	return s1_start_times


def get_file_metadata_human_ph2022(path: str, anno: pd.DataFrame, dataset_object: Physionet2022) -> dict:
	meta = get_base_metadata(path)
	

	filename = Path(path).name
	
		
	meta[const.META_AUDIO_PATH] = path.replace(dataset_object.dataset_path + "\\","")
	patient_id = Path(path).name.split(".")[0].split("_")[0] # first block till _ from file name
	patient_id = int(patient_id)
	#if patient_id in anno.get("Additional ID").values:
	#	patient_id = patient_id + "+" + anno[anno.get("Additional ID") == patient_id].index[0]
	# TODO beachten dass additional ID nur einseitig beschrieben wird -> map oder so als vergleich, beide Fälle müssen gleiche ID haben
	meta[const.META_PATIENT_ID] = patient_id
	meta[const.META_FILENAME] = filename
	meta[const.META_DATASET] = "physionet2022-"+dataset_object.folder_name
	meta[const.META_DIAGNOSIS] = "not-supported-in-dataset"	
	meta[const.META_QUALITY] = 1
	meta[const.META_HEARTCYCLES] = _get_heartcycle_indicies_2022(file_path=path)



	
	#dataclass = anno.loc[id]['Murmur'].squeeze()
	# find the class in the annotation, by collumn "Patient ID"
	dataclass = anno.get("Murmur").loc[patient_id]

	if dataclass == "Absent":
		dataclass = const.CLASS_NEGATIVE
	elif dataclass == "Present":
		dataclass = const.CLASS_POSITIVE
	elif dataclass == "Unknown":
		dataclass = const.CLASS_UNKNOWN
	else:
		raise ValueError(f"Unknown class {dataclass}")
	meta[const.META_LABEL_1] = dataclass
	return meta
# TODO addidional ID beachten

@staticmethod
def parse_physionet2022():
	#https://moody-challenge.physionet.org/2022/#data-table
	print("Start parsing Physionet 2022")
	dataset = Physionet2022()
	dataset.set_run(Run())
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
		#if "50782_MV_1.wav" in file:
			#print("Found 50782_MV_1.wav\n Skipping it for now because no heartcycles are available")
			#continue
		meta = get_file_metadata_human_ph2022(file, annotation, dataset)
		if meta[const.META_LABEL_1] == const.CLASS_UNKNOWN:
			continue # TODO removed unknowns
		pbar.update(1)
		metadata.append(meta)
		
	metadata_df = pd.DataFrame(metadata)
	print(metadata_df.head())
	print("Train path", dataset.meta_file_train)
	# Beim Speichern
	metadata_df[const.META_HEARTCYCLES] = metadata_df[const.META_HEARTCYCLES].apply(json.dumps)
	metadata_df.to_csv(dataset.meta_file_train, index=True, index_label="id", encoding='utf-8')

def dataset_statistics(dataset: AudioDataset):
	dataset.set_run(Run())
	train_data = dataset.load_file_list()
	print("Train data count", len(train_data));print("\n")
	print("Classes count", train_data["label_1"].value_counts());print("\n")
	print("Dataset count", train_data["dataset"].value_counts());print("\n")
	print("Quality count", train_data["quality"].value_counts());print("\n")
	print("SR count", train_data[const.META_SAMPLERATE].value_counts());print("\n")
	print("Channels count", train_data["channels"].value_counts());print("\n")
	print("Length count", train_data["length"].value_counts());print("\n")
	print("Bits count", train_data["bits"].value_counts());print("\n")
	# todo get hearttcycles, statistics of 
	if isinstance(dataset, Physionet2022):
		heartcycles = train_data[const.META_HEARTCYCLES]
		heartcycles_lengths = train_data[const.META_HEARTCYCLES].apply(len)
		print("min heartcycles", train_data[const.META_HEARTCYCLES].apply(lambda x: len(x)).min())
		print("max heartcycles", heartcycles.apply(lambda x: len(x)).max())
		print("mean heartcycles", heartcycles.apply(lambda x: len(x)).mean())
		print("std heartcycles", heartcycles.apply(lambda x: len(x)).std())
		print("median heartcycles", heartcycles.apply(lambda x: len(x)).median())
		print("unique heartcycles", heartcycles.apply(lambda x: len(set(x))).value_counts())
		print("Rows with 0 heartcycles\n", train_data[heartcycles.apply(lambda x: len(x) == 0)]["filename"])
		print("Rows with less than 7 heartcycles\n", train_data[heartcycles.apply(lambda x: len(x) <7)]["filename"])
		

		fig, ax = plt.subplots(figsize=(10, 6))

		# Erstellen des Histogramms mit den Längen
		ax.hist(heartcycles_lengths, bins=100, color='skyblue', edgecolor='black')
		ax.set_title('Verteilung der Herzzyklenanzahl pro Aufnahme', fontsize=15)
		ax.set_xlabel('Anzahl der Herzzyklen pro Aufnahme', fontsize=12)
		ax.set_ylabel('Häufigkeit', fontsize=12)
		ax.set_xticks(range(0, 100, 5))
		ax.grid(True)  # Gitternetzlinien hinzufügen


		# Angenommen, 'length' ist in Samples und 'sr' (Sample Rate) in Hz ist bekannt
		heartcycles_counts = train_data[const.META_HEARTCYCLES].apply(len)
		length_in_seconds = train_data[const.META_LENGTH] / train_data[const.META_SAMPLERATE]
		correlation = heartcycles_counts.corr(length_in_seconds)
		print("Korrelation zwischen der Anzahl der Herzzyklen und der Länge der Aufnahmen:", correlation)
		plt.figure(figsize=(10, 6))
		plt.scatter(length_in_seconds, heartcycles_counts, alpha=0.5)
		plt.title('Korrelation zwischen Anzahl der Herzzyklen und Länge der Aufnahmen')
		plt.xlabel('Länge der Aufnahmen (Sekunden)')
		plt.ylabel('Anzahl der Herzzyklen')
		plt.grid(True)

		# Angenommen, die Funktion calculate_bpm wurde bereits definiert
		train_data['bpm'] = train_data[const.META_HEARTCYCLES].apply(calculate_bpm)

		# Sortieren des DataFrames nach BPM in absteigender Reihenfolge und Anzeigen der Top 5
		top_bpm_records = train_data.sort_values(by='bpm', ascending=False).head(5)
		print(top_bpm_records[['filename', 'bpm']])

		# Anwenden der Funktion auf jede Aufnahme
		bpm_values = train_data[const.META_HEARTCYCLES].apply(calculate_bpm)
		bpm_values_clean = bpm_values.dropna()
		plt.figure(figsize=(10, 6))
		plt.hist(bpm_values_clean, bins=50, color='skyblue', edgecolor='black')
		plt.title('Verteilung der Beats Per Minute (BPM)')
		plt.xlabel('Beats Per Minute')
		plt.ylabel('Häufigkeit')
		plt.grid(True)

		print("Minimale BPM:", bpm_values.min())
		print("Maximale BPM:", bpm_values.max())
		print("Durchschnittliche BPM:", bpm_values.mean())


		plt.show()




def calculate_bpm(data):
	# Extrahiert den ersten und letzten Timestamp der Herzzyklen
	if len(data) > 1:
		start_time = data[0]
		end_time = data[-1]
		duration_in_minutes = (end_time - start_time) / 60
		return len(data) / duration_in_minutes if duration_in_minutes > 0 else None
	return None

def plot_statistics(dataset: AudioDataset):
	import matplotlib.pyplot as plt
	import seaborn as sns
	from mpl_toolkits.axes_grid1 import ImageGrid
	import io
	dataset.set_run(Run())
	data = dataset.load_file_list()
	p_size=12
	font_size=26

	def make_plots(plot_data, dataset: AudioDataset, name=""):
		fig, axs = plt.subplots(1, 3, figsize=(p_size*3, p_size))
		seconds = plot_data["length"] / dataset.target_samplerate
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

def start_parse():
	#parse_physionet2016()
	parse_physionet2022()
	print("Done parsing")

def start_statistics():
	#dataset_statistics(Physionet2016())
	dataset_statistics(Physionet2022())
	
def start_plot():
	#plot_statistics(Physionet2016())
	plot_statistics(Physionet2022())

if __name__ == '__main__':
	# parse folder structure and create meta files
	
	start_parse()
	start_statistics()
	#start_plot()