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

def save_training_data_to_csv(metadata, dataset: AudioDataset):
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

import numpy as np
import pandas as pd

def safe_numeric_conversion(series):
	try:
		return pd.to_numeric(series, errors='coerce')
	except:
		return pd.Series([np.nan] * len(series))

def safe_correlation(s1, s2):
	try:
		s1_numeric = safe_numeric_conversion(s1)
		s2_numeric = safe_numeric_conversion(s2)
		return s1_numeric.corr(s2_numeric)
	except:
		return "N/A (correlation calculation failed)"

def save_statistics(dataset: AudioDataset, stats_dir: Path):
	train_data = dataset.load_file_list()
	dataset_name = dataset.__class__.__name__
	stats_file = stats_dir / f"{dataset_name}_statistics.txt"

	def write_section(f, title):
		f.write(f"\n{'-'*80}\n{title}\n{'-'*80}\n")

	with open(stats_file, "w") as f:
		write_section(f, f"Statistics for {dataset_name}")
		f.write(f"Total count: {len(train_data)}\n")

		write_section(f, "Class Distribution")
		class_counts = train_data[const.META_LABEL_1].value_counts()
		f.write(class_counts.to_string() + "\n\n")
		for label, count in class_counts.items():
			f.write(f"Class {label} percentage: {count/len(train_data)*100:.2f}%\n")

		write_section(f, "Patient Audio File Count")
		patient_file_counts = train_data[const.META_PATIENT_ID].value_counts()
		f.write(f"Unique patients: {len(patient_file_counts)}\n")
		f.write(f"Average files per patient: {patient_file_counts.mean():.2f}\n")
		f.write(f"Median files per patient: {patient_file_counts.median():.2f}\n")
		f.write(f"Min files per patient: {patient_file_counts.min()}\n")
		f.write(f"Max files per patient: {patient_file_counts.max()}\n\n")
		f.write("Distribution of file counts:\n")
		f.write(patient_file_counts.value_counts().sort_index().to_string())
		f.write("\n\n")

		def write_patient_details(dataset: AudioDataset, stats_dir: Path):
			train_data = dataset.load_file_list()
			dataset_name = dataset.__class__.__name__
			details_file = stats_dir / f"{dataset_name}_patient_details.txt"

			patient_file_counts = train_data[const.META_PATIENT_ID].value_counts()

			with open(details_file, "w") as f:
				f.write("Patients with more than 4 data entries:\n")
				for count in sorted(patient_file_counts[patient_file_counts > 4].unique(), reverse=True):
					patients = patient_file_counts[patient_file_counts == count].index
					f.write(f"\nPatients with {count} entries:\n")
					for patient in patients:
						patient_data = train_data[train_data[const.META_PATIENT_ID] == patient]
						f.write(f"  Patient {patient}:\n")
						for idx, row in patient_data.iterrows():
							if const.META_FILENAME in row:
								f.write(f"    {row[const.META_FILENAME]}\n")
							else:
								f.write(f"    Entry ID: {idx}\n")
						f.write("\n")  # Add a blank line between patients

			print(f"Patient details saved to {details_file}")

		# Fügen Sie diese Zeile am Ende Ihrer save_statistics Funktion hinzu:
		write_patient_details(dataset, stats_dir)

		write_section(f, "Audio Length Statistics")
		seconds = safe_numeric_conversion(train_data[const.META_LENGTH] / dataset.target_samplerate)
		total_duration = seconds.sum()
		f.write(f"Total duration: {total_duration:.2f} seconds ({total_duration/3600:.2f} hours)\n")

		for stat in ["mean", "std", "median", "min", "max"]:
			f.write(f"{stat.capitalize()} length (All): {getattr(seconds, stat)():.2f} seconds\n")
			for label in [0, 1]:
				class_seconds = seconds[train_data[const.META_LABEL_1] == label]
				f.write(f"{stat.capitalize()} length (Class {label}): {getattr(class_seconds, stat)():.2f} seconds\n")

		f.write(f"Correlation class<->length: {safe_correlation(train_data[const.META_LABEL_1], seconds)}\n")

		write_section(f, "Extreme Cases")
		for extreme, func in [("longest", "nlargest"), ("shortest", "nsmallest")]:
			f.write(f"15 {extreme} files:\n")
			for idx, row in getattr(train_data, func)(15, const.META_LENGTH).iterrows():
				filename = row.get(const.META_FILENAME, idx)
				f.write(f"{filename}: {row[const.META_LENGTH]/dataset.target_samplerate:.2f} seconds\n")
			f.write("\n")

		write_section(f, "Demographic and Clinical Data")
		for column in [const.META_AGE, const.META_SEX, const.META_HEIGHT, const.META_WEIGHT, const.META_PREGNANT, const.META_DIAGNOSIS]:
			if column in train_data.columns:
				f.write(f"{column} statistics:\n")
				value_counts = train_data[column].value_counts()
				f.write(f"Unique values: {value_counts.count()}\n")
				f.write(f"Top 10 most common:\n{value_counts.head(n=10).to_string()}\n")
				f.write(f"Percentage of rows with these missing values: {train_data[column].isnull().mean()*100:.2f}%\n")

				numeric_data = safe_numeric_conversion(train_data[column])
				if not numeric_data.isnull().all():
					f.write(f"Mean: {numeric_data.mean():.2f}\n")
					f.write(f"Std: {numeric_data.std():.2f}\n")
					f.write(f"Correlation with class: {safe_correlation(train_data[const.META_LABEL_1], numeric_data)}\n")
				else:
					f.write("Non-numeric data, skipping mean, std, and correlation.\n")
				f.write("\n")

		if isinstance(dataset, Physionet2022):
			write_section(f, "Heartcycles Statistics")
			heartcycles = train_data[const.META_HEARTCYCLES]
			heartcycles_counts = heartcycles.apply(len)
			f.write(heartcycles_counts.describe().to_string() + "\n\n")

			f.write("Correlation with audio length: ")
			f.write(f"{safe_correlation(heartcycles_counts, seconds)}\n\n")

			write_section(f, "BPM Statistics")
			try:
				train_data["bpm"] = train_data[const.META_HEARTCYCLES].apply(calculate_bpm)
				bpm_numeric = safe_numeric_conversion(train_data["bpm"])
				f.write(bpm_numeric.describe().to_string() + "\n\n")

				f.write("Correlation BPM with:\n")
				for col in [const.META_LABEL_1, const.META_AGE, "bpm"]:
					if col in train_data.columns:
						f.write(f"- {col}: {safe_correlation(bpm_numeric, train_data[col])}\n")

				for extreme, func in [("slowest", "nsmallest"), ("fastest", "nlargest")]:
					f.write(f"\n10 {extreme} BPM files:\n")
					extreme_bpm = getattr(bpm_numeric, func)(10)
					for idx, bpm in extreme_bpm.items():
						filename = train_data.loc[idx, const.META_FILENAME] if const.META_FILENAME in train_data.columns else f"Index: {idx}"
						f.write(f"{filename}: {bpm:.2f} BPM\n")
			except Exception as e:
				f.write(f"Error calculating BPM statistics: {str(e)}\n")

	print(f"Statistics saved to {stats_file}")
	return train_data

def plot_class_distribution(data, ax, colors):
	class_counts = data[const.META_LABEL_1].value_counts()
	ax.pie(class_counts, labels=["Normal", "Abnormal"], colors=colors, autopct="%1.1f%%", 
		   textprops={"fontsize": 14, "fontweight": "bold"})
	ax.set_title("Class Distribution", fontsize=24, fontweight="bold")

def plot_audio_length_distribution(data, seconds, ax, colors):
	sns.histplot(data=data, x=seconds, hue=data[const.META_LABEL_1].map({0: "Normal", 1: "Abnormal"}),
				 palette=colors, multiple="stack", kde=False, ax=ax, bins=100)
	ax.set_title("Audio Length Distribution", fontsize=24, fontweight="bold")
	ax.set_xlabel("Length (seconds)", fontsize=18)
	ax.set_ylabel("Count", fontsize=18)
	ax.tick_params(axis="both", which="major", labelsize=15)
	ax.legend(title="Class", labels=["Normal", "Abnormal"], fontsize=14, title_fontsize=16)

def plot_bpm_distribution(data, ax, colors):
	sns.histplot(data=data, x="bpm", hue=data[const.META_LABEL_1].map({0: "Normal", 1: "Abnormal"}),
				 palette=colors, multiple="stack", kde=False, ax=ax, bins=100)
	ax.set_title("BPM Distribution", fontsize=24, fontweight="bold")
	ax.set_xlabel("Beats Per Minute", fontsize=18)
	ax.set_ylabel("Count", fontsize=18)
	ax.legend(title="Class", labels=["Normal", "Abnormal"], fontsize=14, title_fontsize=16)
	ax.tick_params(axis="both", which="major", labelsize=14)

def generate_text_content(data, seconds):
	class_counts = data[const.META_LABEL_1].value_counts()
	text_content = f"Total samples: {len(data)}\n"
	text_content += f"Normal: {class_counts[0]} ({class_counts[0]/len(data)*100:.1f}%)\n"
	text_content += f"Abnormal: {class_counts[1]} ({class_counts[1]/len(data)*100:.1f}%)\n\n"
	text_content += f"Audio Length Statistics:\n"
	text_content += f"  Average (All): {seconds.mean():.2f}s\n"
	text_content += f"  Average (Normal): {seconds[data[const.META_LABEL_1] == 0].mean():.2f}s\n"
	text_content += f"  Average (Abnormal): {seconds[data[const.META_LABEL_1] == 1].mean():.2f}s\n"
	text_content += f"  Median: {seconds.median():.2f}s\n"
	text_content += f"  Minimum: {seconds.min():.2f}s\n"
	text_content += f"  Maximum: {seconds.max():.2f}s\n"
	text_content += f"  Correlation (class vs length): {data[const.META_LABEL_1].corr(seconds):.4f}\n"
	return text_content

def generate_bpm_text_content(data):
	text_content = "BPM Statistics:\n"
	text_content += f"  Average (All): {data['bpm'].mean():.2f}\n"
	text_content += f"  Average (Normal): {data[data[const.META_LABEL_1] == 0]['bpm'].mean():.2f}\n"
	text_content += f"  Average (Abnormal): {data[data[const.META_LABEL_1] == 1]['bpm'].mean():.2f}\n"
	text_content += f"  Median: {data['bpm'].median():.2f}\n"
	text_content += f"  Minimum: {data['bpm'].min():.2f}\n"
	text_content += f"  Maximum: {data['bpm'].max():.2f}\n"
	return text_content

def plot_statistics(dataset: AudioDataset, stats_dir: Path):
	data = dataset.load_file_list()
	dataset_name = dataset.__class__.__name__
	seconds = data[const.META_LENGTH] / dataset.target_samplerate
	colors = ["green", "red"]
	
	# Einzelne Plots
	fig, ax = plt.subplots(figsize=(10, 8))
	plot_class_distribution(data, ax, colors)
	plt.savefig(stats_dir / f"{dataset_name}_class_distribution.png", dpi=300, bbox_inches="tight")
	plt.close(fig)

	fig, ax = plt.subplots(figsize=(10, 8))
	plot_audio_length_distribution(data, seconds, ax, colors)
	plt.savefig(stats_dir / f"{dataset_name}_audio_length_distribution.png", dpi=300, bbox_inches="tight")
	plt.close(fig)

	if isinstance(dataset, Physionet2022):
		data["bpm"] = data[const.META_HEARTCYCLES].apply(calculate_bpm)
		fig, ax = plt.subplots(figsize=(10, 8))
		plot_bpm_distribution(data, ax, colors)
		plt.savefig(stats_dir / f"{dataset_name}_bpm_distribution.png", dpi=300, bbox_inches="tight")
		plt.close(fig)

	# Gesamtübersicht
	sns.set_style("whitegrid")
	fig = plt.figure(figsize=(20, 16))
	gs = fig.add_gridspec(3, 2, height_ratios=[2.8, 1.8, 2.8], hspace=0.3)

	fig.suptitle(f"Dataset Statistics for {dataset_name}", fontsize=28, fontweight="bold", y=0.98)

	ax1 = fig.add_subplot(gs[0, 0])
	ax1.set_position([0.15, 0.68, 0.3, 0.22])
	plot_class_distribution(data, ax1, colors)

	ax2 = fig.add_subplot(gs[0, 1])
	ax2.set_position([0.55, 0.68, 0.4, 0.22])
	plot_audio_length_distribution(data, seconds, ax2, colors)

	ax3 = fig.add_subplot(gs[1, 0])
	ax4 = fig.add_subplot(gs[1, 1])
	ax3.set_position([0.15, 0.40, 0.3, 0.22])
	ax4.set_position([0.55, 0.40, 0.4, 0.22])

	text_content1 = generate_text_content(data, seconds)
	ax3.text(0.15, 1.0, text_content1, fontsize=19, ha="left", va="top", transform=ax3.transAxes)
	ax3.axis("off")

	text_content2 = ""
	if isinstance(dataset, Physionet2022):
		text_content2 = generate_bpm_text_content(data)

	ax4.text(0.15, 1.0, text_content2, fontsize=19, ha="left", va="top", transform=ax4.transAxes)
	ax4.axis("off")

	ax5 = fig.add_subplot(gs[2, :])
	if isinstance(dataset, Physionet2022):
		ax5.set_position([0.15, 0.1, 0.8, 0.24])
		plot_bpm_distribution(data, ax5, colors)
	else:
		ax5.axis("off")

	plt.savefig(stats_dir / f"{dataset_name}_overall_statistics.png", dpi=300, bbox_inches="tight")
	plt.close(fig)

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
