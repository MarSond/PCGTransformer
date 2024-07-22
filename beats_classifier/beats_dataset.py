import torch
from torch.utils.data import Dataset

from MLHelper import constants as const
from MLHelper.audio import preprocessing
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
			normed_audio = preprocessing.max_abs_normalization(filtered_audio)
		elif self.config[const.NORMALIZATION] == const.NORMALIZATION_ZSCORE:
			normed_audio = preprocessing.zscore_normalization(filtered_audio)

		# Convert to tensor
		audio_tensor = torch.FloatTensor(normed_audio)

		# Ensure it's a 2D tensor (1, num_samples)
		if audio_tensor.dim() == 1:
			audio_tensor = audio_tensor.unsqueeze(0)

		return audio_tensor, class_id, padding_mask


	def set_mode(self, mode):
		if mode in (const.TRAINING, const.VALIDATION):
			self.mode = mode
		else:
			raise ValueError("Mode must be either 'train' or 'validation'")
