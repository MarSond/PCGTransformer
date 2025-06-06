import logging

import torch
from torch import tensor
from torch.utils.data import Dataset

from MLHelper import constants as const
from MLHelper.audio import augmentation, preprocessing
from MLHelper.audio.audioutils import AudioUtil
from run import Run


class CNN_Dataset(Dataset):

	def __init__(self, datalist, run: Run):
		self.datalist = datalist
		self.run = run
		self.config = run.config
		self.cnn_config = run.config[const.CNN_PARAMS]
		self.n_mels = self.cnn_config[const.N_MELS]
		self.hop_length = self.cnn_config[const.HOP_LENGTH]
		self.n_fft = self.cnn_config[const.N_FFT]
		self.top_db = self.cnn_config[const.TOP_DB]
		self.target_samplerate = run.task.dataset.target_samplerate
		self.augmentation_rate = self.config[const.AUGMENTATION_RATE]
		self.mode = None

	def __len__(self):
		return len(self.datalist)

	def handle_instance(self, current_row):
		# Absolute file path of the audio file - concatenate the audio directory with
		# the relative path
		audio_filename = current_row[const.META_AUDIO_PATH]
		frame_start = current_row[const.CHUNK_RANGE_START]
		frame_end = current_row[const.CHUNK_RANGE_END]
		class_id = current_row[self.config[const.LABEL_NAME]]
		chunk_name = audio_filename.name + "#" + str(frame_start) + "-" + str(frame_end)
		self.run.log(f"Loading {chunk_name}", name=const.LOGGER_METADATA, level=logging.DEBUG)
		# print("file and class: ", audio_file, class_id)
		raw_audio, file_sr, padding_mask = AudioUtil.Loading.load_audiofile(audio_filename, \
			start_frame=frame_start, end_frame=frame_end, target_length=self.config[const.CHUNK_DURATION], \
			pad_method=self.config[const.AUDIO_LENGTH_NORM])
		if file_sr != self.target_samplerate:
			raw_audio = preprocessing.resample(raw_audio, file_sr, self.target_samplerate)

		######## Butter filter
		if self.config[const.BUTTERPASS_LOW] != 0 and self.config[const.BUTTERPASS_HIGH] != 0:
			filtered_audio = preprocessing.Filter.butter_bandpass_filter( \
				raw_audio, lowcut=self.config[const.BUTTERPASS_LOW], \
				highcut=self.config[const.BUTTERPASS_HIGH], \
				fs=self.target_samplerate, order=self.config[const.BUTTERPASS_ORDER])
		else:
			filtered_audio = raw_audio

		######## Normalization
		if self.config[const.NORMALIZATION] == const.NORMALIZATION_MAX_ABS:
			normalized_audio = preprocessing.max_abs_normalization(filtered_audio)
		elif self.config[const.NORMALIZATION] == const.NORMALIZATION_ZSCORE:
			normalized_audio = preprocessing.zscore_normalization(filtered_audio)
		elif self.config[const.NORMALIZATION] == const.NORMALIZATION_NONE:
			normalized_audio = filtered_audio
		else:
			raise ValueError(f"Normalization type {self.config[const.NORMALIZATION]} not supported")

		########

		if self.mode != const.VALIDATION:
			_audio_augmentation = \
				augmentation.AudioAugmentation.get_audio_augmentation(p=self.augmentation_rate)
			_sgram_augmentation = \
				augmentation.AudioAugmentation.get_spectrogram_augmentation(p=self.augmentation_rate)

		if self.mode == const.DEMO:
			# return raw waveform, filtered waveform, raw spectrogram, filtered spectrogram
			sgram_raw = self._get_mel(raw_audio, self.target_samplerate)
			sgram_filtered = self._get_mel(normalized_audio, self.target_samplerate)
			audio_augmented = _audio_augmentation(samples=normalized_audio, sample_rate=self.target_samplerate)
			sgram_processed = self._get_mel(audio_augmented, self.target_samplerate)
			sgram_augmented = _sgram_augmentation(magnitude_spectrogram=sgram_processed)
			sgram_final = sgram_augmented

			row_dict = current_row.to_dict()
			row_dict[const.META_AUDIO_PATH] = str(row_dict[const.META_AUDIO_PATH])
			return (
				raw_audio,
				normalized_audio,
				audio_augmented,
				sgram_raw,
				sgram_filtered,
				sgram_augmented,
				row_dict,
				chunk_name
			)

		if self.mode == const.TRAINING :
			# training uses augmentation and signal filtering
			audio_augmented = _audio_augmentation(samples=normalized_audio, sample_rate=file_sr)
			sgram_raw = self._get_mel(normalized_audio, file_sr)
			sgram_augmented = _sgram_augmentation(magnitude_spectrogram=sgram_raw)
			sgram_final = sgram_augmented

		if self.mode == const.VALIDATION:
			# Validation uses no augmentation but signal filtering
			sgram_final = self._get_mel(normalized_audio, file_sr)

		######## after processing

		if not isinstance(sgram_final, torch.Tensor):
			sgram_final = tensor(sgram_final)
		return sgram_final, class_id, padding_mask

	def __getitem__(self, idx):
		current_row = self.datalist.iloc[idx]
		return self.handle_instance(current_row)

	def _get_mel(self, audio, file_sr):
		return AudioUtil.SignalFeatures.get_melspectrogram_torchaudio(audio, sr=file_sr, n_mels=self.n_mels, \
			top_db=self.top_db, n_fft=self.n_fft, hop_length=self.hop_length, device=self.run.device)

	def set_mode(self, mode):
		if mode in (const.TRAINING, const.VALIDATION, const.TASK_TYPE_DEMO):
			self.mode = mode
		else:
			raise ValueError("Mode must be either 'train' or 'validation'")
