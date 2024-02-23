from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset
from ml_helper import augmentation
from ml_helper import AudioUtil
import parameters as cfg
from os.path import join as pjoin
from torch import tensor
import logging
class CNN_Dataset(Dataset):
	def __init__(self, data, run_config):
		self.data = data
		self.run_config = run_config
		self.base_path = run_config["audio_path"]
		self.n_mels = run_config['n_mels']
		self.hop_length = run_config['hop_length']
		self.n_fft = run_config['n_fft']
		self.top_db = run_config['top_db']
		self.samplerate = run_config['samplerate']
		self.target_samplerate = run_config['target_samplerate']
		self.augmentation_rate = run_config['augmentation_rate']
		self.mode = "train"

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		# Absolute file path of the audio file - concatenate the audio directory with
		# the relative path
		audio_file = pjoin(self.base_path, self.data.loc[idx, 'path'])
		frame_start = self.data.loc[idx, 'range_start']
		frame_end = self.data.loc[idx, 'range_end']
		class_id = self.data.loc[idx, self.run_config['label_name']]

		# print("file and class: ", audio_file, class_id)
		raw_audio, file_sr = AudioUtil.load_audiofile(audio_file, start_frame=frame_start, end_frame=frame_end, target_length=self.run_config['seconds'])
		raw_audio = AudioUtil.resample(raw_audio, file_sr, self.target_samplerate)
		if self.mode == "audio":
			sgram_raw = AudioUtil.get_spectrogram(raw_audio, sr=file_sr, n_mels=self.n_mels, top_db=self.top_db, n_fft=self.n_fft, hop_length=self.hop_length, use_librosa=True)
		if file_sr != self.samplerate:
			logging.getLogger(cfg.AUDIO_LOGGER).warning("Sample rate mismatch: {} != {}".format(file_sr, self.samplerate))
		
		if self.run_config['butterpass_low'] != 0 and self.run_config['butterpass_high'] != 0:
			audio = AudioUtil.butter_bandpass_filter(raw_audio, self.run_config['butterpass_low'], self.run_config['butterpass_high'], file_sr, order=4)
		else: audio = raw_audio
		audio = AudioUtil.normalize(audio)
		# dur_aud = AudioUtil.pad_trunc(rechan, self.duration)
		# shift_aud = AudioUtil.time_shift(dur_aud, self.shift_pct)
		
		if self.mode != "validation":
			_audio_augmentation = augmentation.AudioAugmentation.get_audio_augmentation(p=self.augmentation_rate)
			_sgram_augmentation = augmentation.AudioAugmentation.get_spectrogram_augmentation(p=self.augmentation_rate)

		if self.mode == "audio":
			# return raw waveform, filtered waveform, raw spectrogram, filtered spectrogram
		
			audio_augmented = _audio_augmentation(samples=audio, sample_rate=file_sr)
			sgram_processed = AudioUtil.get_spectrogram(audio_augmented, sr=file_sr, n_mels=self.n_mels, top_db=self.top_db, n_fft=self.n_fft, hop_length=self.hop_length, use_librosa=True)
			sgram_final = _sgram_augmentation(magnitude_spectrogram=sgram_processed)
			
			return raw_audio, audio, sgram_raw, sgram_final, class_id.item(), audio_file
		
		if self.mode == "train" :

			audio_augmented = _audio_augmentation(samples=audio, sample_rate=file_sr)
			sgram_raw = AudioUtil.get_spectrogram(audio_augmented, sr=file_sr, n_mels=self.n_mels, top_db=self.top_db, n_fft=self.n_fft, hop_length=self.hop_length, use_librosa=True)
			sgram_final = _sgram_augmentation(magnitude_spectrogram=sgram_raw)
		
		elif self.mode == "validation":
			sgram_final = AudioUtil.get_spectrogram(audio, sr=file_sr, n_mels=self.n_mels, n_fft=self.n_fft, hop_length=self.hop_length, use_librosa=True)

		tensored_sgram = tensor(sgram_final)
		return tensored_sgram, class_id

	def set_mode(self, mode):
		if mode == "train" or mode == "validation" or mode == "audio":
			self.mode = mode
		else:
			raise ValueError("Mode must be either 'train' or 'validation'")
