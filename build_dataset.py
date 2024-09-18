import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torchaudio
from tqdm.auto import tqdm

import MLHelper.constants as const
from MLHelper.dataset import AudioDataset, Physionet2016, Physionet2022
from run import Run

# ruff: noqa: T201, E501

def save_training_data_to_csv(metadata, dataset):
	metadata_df = pd.DataFrame(metadata)
	print(metadata_df.head())
	print(f"Train path: {dataset.meta_file_train}")
	metadata_df.to_csv(dataset.meta_file_train, index=True, index_label=const.META_ID, encoding="utf-8")

def get_base_metadata(path):
	meta = {key: None for key in AudioDataset.columns}
	info = torchaudio.info(path)
	meta[const.META_SAMPLERATE] = info.sample_rate
	meta[const.META_CHANNELS] = info.num_channels
	meta[const.META_LENGTH] = info.num_frames
	meta[const.META_BITS] = info.bits_per_sample
	return meta

def _get_heartcycle_indicies_2016(file_path: str) -> list:
	return []

def get_file_metadata_human_ph2016(
		path: str, anno: pd.DataFrame, data_classes: pd.DataFrame, dataset_object: Physionet2016) -> dict:
	meta = get_base_metadata(path)
	id = Path(path).stem
	dataset_name = str(Path(path).parents[0]).split("\\")[-1]

	try:
		quality = data_classes.loc[id]["quality"].squeeze()
	except KeyError:
		quality = 1
	if np.isnan(quality):
		quality = 1  # Assume quality is okay if no data is given

	try:
		diagnosis = anno.loc[id]["Diagnosis"]
	except KeyError:
		new_id = anno["Original record name"] == id
		val = anno.loc[new_id]["Diagnosis"].squeeze()
		diagnosis = str(val) if len(val) > 0 else "Unknown"

	try:
		dataclass = int(anno.loc[id]["Class (-1=normal 1=abnormal)"].squeeze())
	except KeyError:
		dataclass = int(data_classes.loc[id]["class"].squeeze())
	dataclass = 0 if dataclass == -1 else dataclass

	base_meta = get_base_metadata(path)
	meta = {
		const.META_AUDIO_PATH: str(Path(path).relative_to(dataset_object.dataset_path)),
		const.META_PATIENT_ID: id,
		const.META_FILENAME: Path(path).name,
		const.META_DIAGNOSIS: diagnosis,
		const.META_DATASET: const.PHYSIONET_2016,
		const.META_DATASET_SUBSET: dataset_name,
		const.META_QUALITY: int(quality),
		const.META_HEARTCYCLES: _get_heartcycle_indicies_2016(path),
		const.META_LABEL_1: dataclass
	}

	return {**base_meta, **meta}

def parse_physionet2016():
	dataset = Physionet2016(Run())
	train_data = list(Path(dataset.train_audio_base_folder).rglob(dataset.train_audio_search_pattern))
	train_data = [Path(path).resolve() for path in train_data]
	dataset_names = ["training-a", "training-b", "training-c", "training-d", "training-e", "training-f"]
	name_list = pd.DataFrame()

	for name in dataset_names:
		subset_liste = list(Path(f"{dataset.dataset_path}/audiofiles/train/{name}").rglob("*.wav"))
		print(f"{name} count: {len(subset_liste)}")
		for path_name in subset_liste:
			file_name = Path(path_name).stem
			name_list = pd.concat([name_list, pd.DataFrame([file_name], columns=["name"])], ignore_index=True)

	name_list.set_index("name", inplace=True)
	print(f"Total count: {len(name_list)}")

	anno = pd.read_csv(Path(dataset.dataset_path) / "Online_Appendix_training_set.csv")
	anno.set_index("Challenge record name", inplace=True)

	print(f"All annotations count: {len(anno)}")

	data_classes = pd.read_csv(Path(dataset.dataset_path) / "classes_SQI.csv")
	data_classes.set_index("name", inplace=True)
	print(data_classes.head())
	annotation_classes = anno[["Class (-1=normal 1=abnormal)"]]
	print(annotation_classes.head())
	error = []
	for name, _ in name_list.iterrows():
		if name in annotation_classes.index:
			anno_class = annotation_classes.loc[name].values[0]
			data_c = data_classes.loc[name].get("class")
			if anno_class != data_c:
				print(f"Name {name} found in annotations and classes is not equal. {anno_class} != {data_c}")
		else:
			print(f"Name {name} not found in regular annotations")
			if name in anno.get("Original record name").values:
				orig_class = (anno.loc[anno["Original record name"] == name].get("Class (-1=normal 1=abnormal)").values[0])
				data_c = data_classes.loc[name].get("class")
				print(f"BUT Name {name} found in Original record name with class {orig_class} and is in data_classes with class {data_c}")
			else:
				print(f"Name {name} WAS NOT Original names")
				if name in data_classes.index:
					print(f"Name {name} in data_classes with class {data_classes.loc[name].get('class')}")
					error.append(name)
				else:
					raise ValueError(f"Name {name} WAS NOT in ANYTHING")
	print(f"Len error: {len(error)}")

	print(f"All train data count: {len(train_data)}")
	print(f"All annotations count: {len(anno)}")
	print(f"All classes count: {len(data_classes)}")
	metadata = []

	pbar = tqdm(total=len(train_data), position=0, leave=True, desc="Human data list")
	for file in train_data:
		metadata.append(get_file_metadata_human_ph2016(file, anno, data_classes, dataset))
		pbar.update(1)
	save_training_data_to_csv(metadata, dataset)

def _get_heartcycle_indicies_2022(file_path: str) -> list:
	base_path = Path(file_path).parents[0]
	audio_filename = Path(file_path).name.removesuffix(".wav")
	heartcycle_file = Path(base_path) / f"{audio_filename}.tsv"
	assert Path(heartcycle_file).exists(), f"File {heartcycle_file} does not exist"

	s1_start_times = []

	with Path.open(heartcycle_file) as f:
		lines = f.readlines()

	for line in lines:
		parts = line.strip().split("\t")
		if len(parts) < 3:
			raise ValueError(f"Line {line} does not have 3 parts")

		wave_type = parts[2]
		if wave_type == "1":
			start_time = float(parts[0])
			s1_start_times.append(start_time)

	return s1_start_times

def get_file_metadata_human_ph2022(path: str, anno: pd.DataFrame, dataset_object: Physionet2022) -> dict:
	dataset_path = Path(dataset_object.dataset_path)
	file_name = Path(path).name
	patient_id = int(file_name.split(".")[0].split("_")[0])
	dataclass = anno.get("Murmur").loc[patient_id]

	if dataclass == "Absent":
		dataclass = const.CLASS_NEGATIVE
	elif dataclass == "Present":
		dataclass = const.CLASS_POSITIVE
	elif dataclass == "Unknown":
		dataclass = const.CLASS_UNKNOWN
	else:
		raise ValueError(f"Unknown class {dataclass}")

	base_meta = get_base_metadata(path)
	meta = {
		const.META_AUDIO_PATH: str(Path(path).relative_to(dataset_path)),
		const.META_PATIENT_ID: patient_id,
		const.META_FILENAME: file_name,
		const.META_DATASET: const.PHYSIONET_2022,
		const.META_DATASET_SUBSET: f"{dataset_object.folder_name}",
		const.META_DIAGNOSIS: None,
		const.META_QUALITY: 1,
		const.META_AGE: anno.get("Age").loc[patient_id],
		const.META_SEX: anno.get("Sex").loc[patient_id],
		const.META_HEIGHT : anno.get("Height").loc[patient_id],
		const.META_WEIGHT: anno.get("Weight").loc[patient_id],
		const.META_PREGNANT: anno.get("Pregnancy status").loc[patient_id],
		const.META_HEARTCYCLES: _get_heartcycle_indicies_2022(file_path=path),
		const.META_LABEL_1: dataclass,
		const.META_ADDITIONAL_ID: anno.get("Additional ID").loc[patient_id]
	}
	return {**base_meta, **meta}

def _fix_2022_additional_id(metadata: pd.DataFrame):
	if const.META_ADDITIONAL_ID not in metadata.columns:
		return metadata

	# Filter out rows with missing values
	#metadata_no = metadata[metadata[const.META_ADDITIONAL_ID].notna()]

	# Build mapping between original and additional IDs
	mapping = {}
	for idx, row in metadata.iterrows():
		original_id = row[const.META_PATIENT_ID]
		if const.META_ADDITIONAL_ID not in row or row[const.META_ADDITIONAL_ID] is None or np.isnan(row[const.META_ADDITIONAL_ID]):
			continue
		additional_id = int(row[const.META_ADDITIONAL_ID])

		if original_id not in mapping and additional_id not in mapping:
			new_id = min(original_id, additional_id)
			mapping[original_id] = new_id
			mapping[additional_id] = new_id
		elif original_id in mapping:
			mapping[additional_id] = mapping[original_id]
		elif additional_id in mapping:
			mapping[original_id] = mapping[additional_id]

	# Update Patient IDs based on the mapping
	metadata.loc[:, const.META_PATIENT_ID] = metadata[const.META_PATIENT_ID].map(mapping).fillna(metadata[const.META_PATIENT_ID])

	# Remove the Additional ID column as it's no longer needed
	metadata = metadata.drop(columns=[const.META_ADDITIONAL_ID])

	return metadata

def parse_physionet2022():
	print("Start parsing Physionet 2022")
	dataset = Physionet2022(Run())
	annotation = pd.read_csv(Path(dataset.dataset_path) / "training_data.csv", index_col="Patient ID")
	print(annotation.head())
	print("Annotation types", annotation.dtypes)
	print(f"Looking for train data in {dataset.train_audio_search_pattern}")
	annotation.index = annotation.index.astype(int)
	training_files = list(Path(dataset.train_audio_base_folder).rglob(dataset.train_audio_search_pattern))
	training_files = [Path.resolve(path) for path in training_files]
	print(f"All existing audio files count: {len(training_files)}")
	print(f"All annotations count: {len(annotation)}")
	metadata = []
	pbar = tqdm(total=len(training_files), position=0, leave=True, desc="Physionet 2022 Human data list")
	for file in training_files:
		meta = get_file_metadata_human_ph2022(file, annotation, dataset)
		if meta[const.META_LABEL_1] != const.CLASS_UNKNOWN: # Skip unknown data
			metadata.append(meta)
		pbar.update(1)

	metadata_df = pd.DataFrame(metadata)
	metadata_df = _fix_2022_additional_id(metadata_df)

	print(metadata_df.head())
	print(f"Train path: {dataset.meta_file_train}")
	metadata_df[const.META_HEARTCYCLES] = metadata_df[const.META_HEARTCYCLES].apply(json.dumps)
	metadata_df.to_csv(dataset.meta_file_train, index=True, index_label="id", encoding="utf-8")

def start_parse():
	parse_physionet2016()
	parse_physionet2022()
	print("Done parsing")

	##########

def create_output_directories(dataset):
	base_path = Path(dataset.dataset_path)
	stats_dir = base_path / "statistics"
	stats_dir.mkdir(parents=True, exist_ok=True)
	return stats_dir

def calculate_bpm(data):
	if len(data) > 1:
		start_time = data[0]
		end_time = data[-1]
		duration_in_minutes = (end_time - start_time) / 60
		return len(data) / duration_in_minutes if duration_in_minutes > 0 else None
	return None

def save_statistics(dataset: AudioDataset, stats_dir: Path):
	train_data = dataset.load_file_list()
	dataset_name = dataset.__class__.__name__
	stats_file = stats_dir / f"{dataset_name}_statistics.txt"

	with open(stats_file, "w") as f:
		f.write(f"Statistics for {dataset_name}\n")
		f.write(f"Total count: {len(train_data)}\n\n")

		# Class distribution
		class_counts = train_data[const.META_LABEL_1].value_counts()
		f.write("Class distribution:\n")
		f.write(class_counts.to_string())
		f.write("\n\n")

		# Length statistics
		seconds = train_data[const.META_LENGTH] / train_data[const.META_SAMPLERATE]
		total_duration = seconds.sum()
		f.write(f"Total duration: {total_duration:.2f} seconds ({total_duration/3600:.2f} hours)\n")
		f.write(f"Average length (All): {seconds.mean():.2f} seconds\n")
		f.write(f"Average length (Negative): {seconds[train_data[const.META_LABEL_1] == 0].mean():.2f} seconds\n")
		f.write(f"Average length (Positive): {seconds[train_data[const.META_LABEL_1] == 1].mean():.2f} seconds\n")
		f.write(f"Standard deviation (All): {seconds.std():.2f} seconds\n")
		f.write(f"Standard deviation (Negative): {seconds[train_data[const.META_LABEL_1] == 0].std():.2f} seconds\n")
		f.write(f"Standard deviation (Positive): {seconds[train_data[const.META_LABEL_1] == 1].std():.2f} seconds\n")
		f.write(f"Median length: {seconds.median():.2f} seconds\n")
		f.write(f"Minimum length: {seconds.min():.2f} seconds\n")
		f.write(f"Maximum length: {seconds.max():.2f} seconds\n")
		f.write(f"Correlation class<->length: {train_data[const.META_LABEL_1].corr(seconds):.4f}\n\n")

		# Class percentages
		f.write(f"Negative class percentage: {class_counts.get(0, 0)/len(train_data)*100:.2f}%\n")
		f.write(f"Positive class percentage: {class_counts.get(1, 0)/len(train_data)*100:.2f}%\n\n")

		# 15 longest and shortest files
		f.write("15 longest files:\n")
		for idx, row in train_data.nlargest(15, const.META_LENGTH).iterrows():
			filename = row.get(const.META_FILENAME, idx)
			f.write(f"{filename}: {row[const.META_LENGTH]/dataset.target_samplerate:.2f} seconds\n")
		f.write("\n")

		f.write("15 shortest files:\n")
		for idx, row in train_data.nsmallest(15, const.META_LENGTH).iterrows():
			filename = row.get(const.META_FILENAME, idx)
			f.write(f"{filename}: {row[const.META_LENGTH]/dataset.target_samplerate:.2f} seconds\n")
		f.write("\n")

		# Additional statistics for demographic and clinical data
		for column in [const.META_AGE, const.META_SEX, const.META_HEIGHT,const.META_WEIGHT, const.META_PREGNANT]:
			if column in train_data.columns:
				f.write(f"{column} statistics:\n")
				f.write("count unique top freq\n")
				f.write(train_data[column].value_counts().to_string())
				f.write("\n\n")

				if train_data[column].dtype in ["int6", "float64"]:
					f.write(f"Correlation {column}<->class: {train_data[const.META_LABEL_1].corr(train_data[column]):.4f}\n\n")
		# TODO implement stats
		# # Murmur-related statistics
		# if const.MURMU in train_data.columns:
		# 	f.write("Murmur distribution:\n")
		# 	f.write(train_data[const.META_MURMUR].value_counts().to_string())
		# 	f.write("\n\n")

		# # Recording locations statistics
		# if const.META_RECORDING_LOCATIONS in train_data.columns:
		# 	f.write("Recording locations distribution:\n")
		# 	f.write(train_data[const.META_RECORDING_LOCATIONS].value_counts().to_string())
		# 	f.write("\n\n")

		# # Outcome statistics
		# if const.META_OUTCOME in train_data.columns:
		# 	f.write("Outcome distribution:\n")
		# 	f.write(train_data[const.META_OUTCOME].value_counts().to_string())
		# 	f.write("\n\n")

		# # Campaign statistics
		# if const.META_CAMPAIGN in train_data.columns:
		# 	f.write("Campaign distribution:\n")
		# 	f.write(train_data[const.META_CAMPAIGN].value_counts().to_string())
		# 	f.write("\n\n")

		if isinstance(dataset, Physionet2022):
			heartcycles = train_data[const.META_HEARTCYCLES]
			heartcycles_counts = heartcycles.apply(len)
			f.write("Heartcycles statistics:\n")
			f.write(f"Min: {heartcycles_counts.min()}\n")
			f.write(f"Max: {heartcycles_counts.max()}\n")
			f.write(f"Mean: {heartcycles_counts.mean():.2f}\n")
			f.write(f"Std: {heartcycles_counts.std():.2f}\n")
			f.write(f"Median: {heartcycles_counts.median()}\n\n")

			train_data["bpm"] = train_data[const.META_HEARTCYCLES].apply(calculate_bpm)
			f.write("BPM statistics:\n")
			f.write(train_data["bpm"].describe().to_string())
			f.write("\n\n")

			f.write("10 slowest BPM files:\n")
			for idx, row in train_data.nsmallest(10, "bpm").iterrows():
				filename = row.get(const.META_FILENAME, idx)
				f.write(f"{filename}: {row['bpm']:.2f} BPM\n")
			f.write("\n")

			f.write("10 fastest BPM files:\n")
			for idx, row in train_data.nlargest(10, "bpm").iterrows():
				filename = row.get(const.META_FILENAME, idx)
				f.write(f"{filename}: {row['bpm']:.2f} BPM\n")
			f.write("\n")

	print(f"Statistics saved to {stats_file}")
	return train_data

def plot_statistics(dataset: AudioDataset, stats_dir: Path):
	data = dataset.load_file_list()
	dataset_name = dataset.__class__.__name__

	# Length distribution
	plt.figure(figsize=(12, 8))
	seconds = data[const.META_LENGTH] / dataset.target_samplerate
	sns.histplot(data=seconds, kde=True)
	plt.title(f"{dataset_name} - Audio Length Distribution")
	plt.xlabel("Seconds")
	plt.ylabel("Count")
	plt.savefig(stats_dir / f"{dataset_name}_length_distribution.png")
	plt.close()

	# Class distribution (using the old plot style)
	p_size = 12
	font_size = 16
	fig, axs = plt.subplots(1, 3, figsize=(p_size * 3, p_size))

	# Pie chart
	class_counts = data[const.META_LABEL_1].value_counts()
	axs[0].pie(class_counts, labels=["NORMAL", "ABNORMAL"],
				autopct="%1.1f%%", colors=["green", "red"])
	axs[0].set_title("Class Distribution", fontsize=font_size)

	# Statistics text
	text_content = f"Average length (Negative): {seconds[data[const.META_LABEL_1] == 0].mean():.2f}s\n"
	text_content += f"Average length (Positive): {seconds[data[const.META_LABEL_1] == 1].mean():.2f}s\n"
	text_content += f"Average length (All): {seconds.mean():.2f}s\n"
	text_content += f"Standard deviation (Negative): {seconds[data[const.META_LABEL_1] == 0].std():.2f}s\n"
	text_content += f"Standard deviation (Positive): {seconds[data[const.META_LABEL_1] == 1].std():.2f}s\n"
	text_content += f"Standard deviation (All): {seconds.std():.2f}s\n"
	text_content += f"Median length: {seconds.median():.2f}s\n"
	text_content += f"Minimum length: {seconds.min():.2f}s\n"
	text_content += f"Maximum length: {seconds.max():.2f}s\n"
	text_content += f"Correlation class<->length: {data[const.META_LABEL_1].corr(seconds):.4f}\n"
	text_content += f"Counts: neg.: {class_counts[0]} pos.: {class_counts[1]} total: {len(data)}"

	axs[1].text(0, 0.5, text_content, fontsize=font_size-2, ha="left", va="center")
	axs[1].axis("off")

	# Length histogram
	axs[2].hist(seconds, bins=100, color="blue", alpha=0.7)
	axs[2].set_title("Length Histogram", fontsize=font_size)
	axs[2].set_xlabel("Seconds", fontsize=font_size-2)
	axs[2].set_ylabel("Count", fontsize=font_size-2)

	plt.tight_layout()
	plt.savefig(stats_dir / f"{dataset_name}_class_and_length_distribution.png")
	plt.close()

	if isinstance(dataset, Physionet2022):
		data["bpm"] = data[const.META_HEARTCYCLES].apply(calculate_bpm)
		plt.figure(figsize=(12, 8))
		sns.histplot(data=data["bpm"].dropna(), kde=True)
		plt.title(f"{dataset_name} - BPM Distribution")
		plt.xlabel("Beats Per Minute")
		plt.ylabel("Count")
		plt.savefig(stats_dir / f"{dataset_name}_bpm_distribution.png")
		plt.close()

	plt.figure(figsize=(10, 8))
	corr_data = data[[const.META_LABEL_1, const.META_LENGTH, const.META_SAMPLERATE, const.META_CHANNELS]]
	sns.heatmap(corr_data.corr(), annot=True, cmap="coolwarm")
	plt.title(f"{dataset_name} - Correlation Heatmap")
	plt.savefig(stats_dir / f"{dataset_name}_correlation_heatmap.png")
	plt.close()

	print(f"Plots saved in {stats_dir}")

def generate_statistics_and_plots(dataset: AudioDataset):
	stats_dir = create_output_directories(dataset)
	data = save_statistics(dataset, stats_dir)
	plot_statistics(dataset, stats_dir)

def start_statistics_and_plots():
	generate_statistics_and_plots(Physionet2016(Run()))
	generate_statistics_and_plots(Physionet2022(Run()))
	print("Done generating statistics and plots")

if __name__ == "__main__":
	start_parse()
	start_statistics_and_plots()
