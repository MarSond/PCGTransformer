from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset
from MLHelper.audio import augmentation, preprocessing
from MLHelper.audio.audioutils import AudioUtil

from os.path import join as pjoin
import torch
from torch import tensor
from run import Run
from MLHelper import constants as const

class CNN_Dataset(Dataset):

	def __init__(self, datalist, run: Run):
		self.datalist = datalist
		self.run = run
		self.config = run.config
		self.cnn_config = run.config[const.CNN_PARAMS]
		self.base_path = run.task.dataset.dataset_path
		self.n_mels = self.cnn_config[const.N_MELS]
		self.hop_length = self.cnn_config[const.HOP_LENGTH]
		self.n_fft = self.cnn_config[const.N_FTT]
		self.top_db = self.cnn_config[const.TOP_DB]
		self.target_samplerate = run.task.dataset.target_samplerate
		self.augmentation_rate = self.config[const.AUGMENTATION_RATE]
		self.mode = None

	def __len__(self):
		return len(self.datalist)
		# TODO verify datalist is only data of current fold
  
	def __getitem__(self, idx):
		# Absolute file path of the audio file - concatenate the audio directory with
		# the relative path
		current_row = self.datalist.iloc[idx]
		audio_file = pjoin(self.base_path, current_row['path'])
		frame_start = current_row['range_start']
		frame_end = current_row['range_end']
		class_id = current_row[self.config[const.LABEL_NAME]]

		# print("file and class: ", audio_file, class_id)
		raw_audio, file_sr = AudioUtil.Loading.load_audiofile(audio_file, start_frame=frame_start, end_frame=frame_end, target_length=self.config[const.CHUNK_DURATION])
		if file_sr != self.target_samplerate:
			self.run.logger_dict["preprocessing"].info("Sample rate mismatch: {} != {}".format(file_sr, self.target_samplerate))
			raw_audio = preprocessing.resample(raw_audio, file_sr, self.target_samplerate)
		
		if self.mode == "audio":
			sgram_raw = AudioUtil.SignalFeatures.get_melspectrogram(raw_audio, sr=file_sr, n_mels=self.n_mels, top_db=self.top_db, n_fft=self.n_fft, hop_length=self.hop_length, use_librosa=True)

		if self.cnn_config[const.BUTTERPASS_LOW] != 0 and self.cnn_config[const.BUTTERPASS_HIGH] != 0:
			audio = preprocessing.Filter.butter_bandpass_filter(raw_audio, self.cnn_config[const.BUTTERPASS_LOW], self.cnn_config[const.BUTTERPASS_HIGH], file_sr, order=self.cnn_config[const.BUTTERPASS_ORDER])
		else: audio = raw_audio
		audio = preprocessing.max_abs_normalization(audio)
		# dur_aud = AudioUtil.pad_trunc(rechan, self.duration)
		# shift_aud = AudioUtil.time_shift(dur_aud, self.shift_pct)

		if self.mode != const.VALIDATION:
			_audio_augmentation = augmentation.AudioAugmentation.get_audio_augmentation(p=self.augmentation_rate)
			_sgram_augmentation = augmentation.AudioAugmentation.get_spectrogram_augmentation(p=self.augmentation_rate)

		if self.mode == const.MODE_AUDIO_VISUALIZATION:
			# return raw waveform, filtered waveform, raw spectrogram, filtered spectrogram
		
			audio_augmented = _audio_augmentation(samples=audio, sample_rate=file_sr)
			sgram_processed = AudioUtil.SignalFeatures.get_melspectrogram(audio_augmented, sr=file_sr, n_mels=self.n_mels, top_db=self.top_db, n_fft=self.n_fft, hop_length=self.hop_length)
			sgram_final = _sgram_augmentation(magnitude_spectrogram=sgram_processed)
			
			return raw_audio, audio, sgram_raw, sgram_final, class_id.item(), audio_file
		
		if self.mode == const.TRAINING :

			audio_augmented = _audio_augmentation(samples=audio, sample_rate=file_sr)
			sgram_raw = AudioUtil.SignalFeatures.get_melspectrogram(audio_augmented, sr=file_sr, n_mels=self.n_mels, top_db=self.top_db, n_fft=self.n_fft, hop_length=self.hop_length)
			sgram_final = _sgram_augmentation(magnitude_spectrogram=sgram_raw)	
		elif self.mode == const.VALIDATION:
			sgram_final = AudioUtil.SignalFeatures.get_melspectrogram(audio, sr=file_sr, n_mels=self.n_mels, n_fft=self.n_fft, hop_length=self.hop_length)
		else:
			raise ValueError("Mode must be either 'train' or 'validation'")
		if not isinstance(sgram_final, torch.Tensor):
			sgram_final = tensor(sgram_final)
		return sgram_final, class_id

	def set_mode(self, mode):
		if mode == const.TRAINING or mode == const.VALIDATION or mode == const.MODE_AUDIO_VISUALIZATION:
			self.mode = mode
		else:
			raise ValueError("Mode must be either 'train' or 'validation'")
