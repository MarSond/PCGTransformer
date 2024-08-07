import logging

import torch
from torch.utils.data import Dataset

from MLHelper import constants as const
from MLHelper.audio import augmentation, preprocessing
from MLHelper.audio.audioutils import AudioUtil
from run import Run


class BEATsDataset(Dataset):
	def __init__(self, datalist, run: Run):
		self.datalist = datalist
		self.run = run
		self.config = run.config
		self.beats_config = run.config[const.TRANSFORMER_PARAMS]
		self.target_samplerate = run.task.dataset.target_samplerate
		self.chunk_duration = self.config[const.CHUNK_DURATION]
		self.augmentation_rate = self.config[const.AUGMENTATION_RATE]
		self.mode = None

	def __len__(self):
		return len(self.datalist)

	def __getitem__(self, idx):
		current_row = self.datalist.iloc[idx]
		return self.handle_instance(current_row)

	def handle_instance(self, current_row):
		audio_filename = current_row[const.META_AUDIO_PATH]
		frame_start = current_row[const.CHUNK_RANGE_START]
		frame_end = current_row[const.CHUNK_RANGE_END]
		class_id = current_row[self.config[const.LABEL_NAME]]
		chunk_name = audio_filename.name + "#" + str(frame_start) + "-" + str(frame_end)
		self.run.log(f"Loading {chunk_name}", name=const.LOGGER_METADATA, level=logging.INFO)

		raw_audio, file_sr, padding_mask = AudioUtil.Loading.load_audiofile(
			audio_filename,
			start_frame=frame_start,
			end_frame=frame_end,
			target_length=self.chunk_duration,
			pad_method=self.config[const.AUDIO_LENGTH_NORM]
		)

		resampled_audio = preprocessing.resample(raw_audio, file_sr, 16000) # resample to BEATs samplerate

		if self.config[const.BUTTERPASS_LOW] != 0 and self.config[const.BUTTERPASS_HIGH] != 0:
			filtered_audio = preprocessing.Filter.butter_bandpass_filter( \
				resampled_audio, lowcut=self.config[const.BUTTERPASS_LOW], \
				highcut=self.config[const.BUTTERPASS_HIGH], \
				fs=self.target_samplerate, order=self.config[const.BUTTERPASS_ORDER])
		else:
			filtered_audio = resampled_audio

		# Normalize
		if self.config[const.NORMALIZATION] == const.NORMALIZATION_MAX_ABS:
			normalized_audio = preprocessing.max_abs_normalization(filtered_audio)
		elif self.config[const.NORMALIZATION] == const.NORMALIZATION_ZSCORE:
			normalized_audio = preprocessing.zscore_normalization(filtered_audio)

		if self.mode != const.VALIDATION:
			_audio_augmentation = \
				augmentation.AudioAugmentation.get_audio_augmentation(p=self.augmentation_rate)

		if self.mode == const.TRAINING :
			# training uses augmentation and signal filtering
			audio_augmented = _audio_augmentation(samples=normalized_audio, sample_rate=file_sr)

		if self.mode == const.VALIDATION:
			# Validation uses no augmentation but signal filtering
			audio_augmented = normalized_audio

		if self.mode == const.DEMO:
			audio_augmented = _audio_augmentation(samples=normalized_audio, sample_rate=self.target_samplerate)

			row_dict = current_row.to_dict()
			row_dict[const.META_AUDIO_PATH] = str(row_dict[const.META_AUDIO_PATH])
			return (
				raw_audio,
				normalized_audio,
				audio_augmented,
				row_dict,
				chunk_name
			)

		# Convert to tensor
		audio_tensor = torch.FloatTensor(audio_augmented)

		# Ensure it's a 2D tensor (1, num_samples)
		if audio_tensor.dim() == 1:
			audio_tensor = audio_tensor.unsqueeze(0)

		return audio_tensor, class_id, padding_mask


	def set_mode(self, mode):
		if mode in (const.TRAINING, const.VALIDATION, const.TASK_TYPE_DEMO):
			self.mode = mode
		else:
			raise ValueError("Mode must be either 'train' or 'validation'")
