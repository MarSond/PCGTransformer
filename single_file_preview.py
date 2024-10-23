from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch

import project_config
from cnn_classifier.cnn_dataset import CNN_Dataset
from MLHelper.audio import audioutils, preprocessing
from MLHelper.constants import *
from MLHelper.dataset import Physionet2022


def compare_audio_processing(audio_file: Path, config: dict):
	# Erstelle ein vereinfachtes Run-ähnliches Objekt
	class SimpleRun:

		class DemoTask:
			def __init__(self, dataset):
				self.dataset = dataset

		def __init__(self, config):
			self.config = config
			self.logger_dict = {LOGGER_METADATA: self.DummyLogger(),
			LOGGER_TENSOR: self.DummyLogger(),
			LOGGER_TRAINING: self.DummyLogger(),
			LOGGER_PREPROCESSING: self.DummyLogger(),}
			self.device = torch.device("cuda")

		def set_task(self, dataset):
			self.task = self.DemoTask(dataset)
		class DummyLogger:
			def info(self, *args, **kwargs):
				pass  # Dummy info-Funktion
			def error(self, *args, **kwargs):
				pass  # Dummy error-Funktion
			def warning(self, *args, **kwargs):
				pass  # Dummy warning-Funktion
			def debug(self, *args, **kwargs):
				pass  # Dummy debug-Funktion
			def warn(self, *args, **kwargs):
				pass  # Dummy warn-Funktion

		def log_training(self, *args, **kwargs):
			pass  # Dummy log-Funktion


		def log(self, *args, **kwargs):
			pass  # Dummy log-Funktion

	simple_run = SimpleRun(config)

	# Erstelle und initialisiere das Dataset
	dataset = Physionet2022(simple_run)
	dataset.load_file_list()
	dataset.prepare_chunks()
	simple_run.set_task(dataset)

	# Finde die entsprechende Zeile in dataset.chunk_list
	chunk_data = dataset.chunk_list[dataset.chunk_list[META_AUDIO_PATH].apply(lambda x: Path(x).name) == audio_file.name].iloc[0]

	# Erstelle ein CNN_Dataset-Objekt
	cnn_dataset = CNN_Dataset(dataset.chunk_list, simple_run)
	#cnn_dataset.target_samplerate = 2000  # Setze manuell für Physionet2022
	cnn_dataset.set_mode(TASK_TYPE_DEMO)

	# Hole die Daten im Demo-Modus
	raw_audio, normalized_audio, augmented_audio, sgram_raw, sgram_filtered, sgram_augmented, row_dict, chunk_name = cnn_dataset.handle_instance(chunk_data)

	cycle_markers = row_dict[META_HEARTCYCLES]

	fig, axes = plt.subplots(3, 2, figsize=(20, 16))
	plt.subplots_adjust(hspace=0.2, wspace=0.4)
	fig.suptitle(f"Audio Processing Comparison - {chunk_name} class: {row_dict[META_LABEL_1]}", fontsize=16)

	sr = cnn_dataset.target_samplerate

	# Linke Spalte: Signale
	audioutils.AudioUtil.SignalPlotting.show_signal(samples=raw_audio, samplerate=sr, ax=axes[0, 0], cycle_marker=cycle_markers, raw=True)
	axes[0, 0].set_title("Original Signal")

	audioutils.AudioUtil.SignalPlotting.show_signal(samples=normalized_audio, samplerate=sr, ax=axes[1, 0], cycle_marker=cycle_markers, raw=True)
	axes[1, 0].set_title(f"Normalized + Filtered Signal")

	audioutils.AudioUtil.SignalPlotting.show_signal(samples=augmented_audio, samplerate=sr, ax=axes[2, 0], cycle_marker=cycle_markers, raw=True)
	axes[2, 0].set_title(f"Augmented Signal ")

	# Rechte Spalte: Spektrogramme
	audioutils.AudioUtil.SignalPlotting.show_mel_spectrogram(
		sgram_raw, cnn_dataset.target_samplerate, ax=axes[0, 1], top_db=config[CNN_PARAMS][TOP_DB], hop_length=config[CNN_PARAMS][HOP_LENGTH]
	)
	axes[0, 1].set_title("Raw Mel Spectrogram")

	audioutils.AudioUtil.SignalPlotting.show_mel_spectrogram(
		sgram_filtered, cnn_dataset.target_samplerate, ax=axes[1, 1], top_db=config[CNN_PARAMS][TOP_DB], hop_length=config[CNN_PARAMS][HOP_LENGTH]
	)
	axes[1, 1].set_title("Filtered Mel Spectrogram")

	audioutils.AudioUtil.SignalPlotting.show_mel_spectrogram(
		sgram_augmented, cnn_dataset.target_samplerate, ax=axes[2, 1], top_db=config[CNN_PARAMS][TOP_DB], hop_length=config[CNN_PARAMS][HOP_LENGTH]
	)
	axes[2, 1].set_title("Augmented Mel Spectrogram")

	plt.tight_layout(rect=[0, 0.03, 1, 0.95])
	plt.show()

	print(f"Audio Chunk Details:")
	print(f"Chunk Name: {chunk_name}")
	print(f"Class ID: {row_dict[config[LABEL_NAME]]}")
	print(f"Sample Rate: {cnn_dataset.target_samplerate}")
	print(f"Duration: {config[CHUNK_DURATION]} seconds")
	print(f"Normalization: {config[NORMALIZATION]}")
	print(f"Butterworth Filter: Low={config[BUTTERPASS_LOW]}, High={config[BUTTERPASS_HIGH]}, Order={config[BUTTERPASS_ORDER]}")
	print(f"Mel Spectrogram: n_mels={config[CNN_PARAMS][N_MELS]}, n_fft={config[CNN_PARAMS][N_FFT]}, hop_length={config[CNN_PARAMS][HOP_LENGTH]}, top_db={config[CNN_PARAMS][TOP_DB]}")

# Beispielverwendung
config = project_config.project_config

config_demo = {
	TASK_TYPE: TASK_TYPE_DEMO,
	KFOLD_SPLITS: 1,
	BATCH_SIZE: 1,
	METADATA_FRAC: 1.0,
	AUGMENTATION_RATE: 1.0,
}

config.update(config_demo)

config_override = {
	TRAIN_DATASET: PHYSIONET_2022,
	NORMALIZATION: NORMALIZATION_MAX_ABS,
	CHUNK_DURATION: 27.0,
	CHUNK_METHOD: CHUNK_METHOD_FIXED,
	AUDIO_LENGTH_NORM: LENGTH_NORM_PADDING,
	# CNN_PARAMS: {
	# 	N_MELS: 128,
	# 	HOP_LENGTH: 512,
	# 	N_FFT: 1024,
	# 	TOP_DB: 80,
	# }
}

config.update(config_override)

audio_file = Path("data/physionet2022/training_data/49824_MV.wav")

compare_audio_processing(audio_file, config)
