import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def plot_spectrogram(y, sr, n_fft, hop_length, n_mels, title):
	S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
	S_dB = librosa.power_to_db(S, ref=np.max)
	plt.figure(figsize=(10, 4))
	librosa.display.specshow(S_dB, x_axis="time", y_axis="mel", sr=sr, hop_length=hop_length)
	plt.colorbar(format="%+2.0f dB")
	plt.title(title)
	plt.tight_layout()

def compare_spectrograms(audio_file, params_list):
	y, sr = librosa.load(audio_file, sr=None)

	fig, axes = plt.subplots(len(params_list), 1, figsize=(12, 4*len(params_list)))
	fig.suptitle("Spectrogram Parameter Comparison", fontsize=16)

	for i, params in enumerate(params_list):
		S = librosa.feature.melspectrogram(y=y, sr=sr, **params)
		S_dB = librosa.power_to_db(S, ref=8000)

		img = librosa.display.specshow(S_dB, x_axis="time", y_axis="mel", sr=sr,
										hop_length=params["hop_length"], ax=axes[i])
		axes[i].set_title(f"n_fft={params["n_fft"]}, hop_length={params["hop_length"]}, n_mels={params["n_mels"]}")

	plt.tight_layout()
	plt.show()

# Example usage
audio_file = "data/physionet2022/training_data/2530_AV.wav"  # Replace with your audio file path

params_list = [
	{"n_fft": 1024, "hop_length": 128, "n_mels": 512},  # Original parameters
	{"n_fft": 512, "hop_length": 64, "n_mels": 256},    # Lower resolution
	{"n_fft": 2048, "hop_length": 256, "n_mels": 1024}, # Higher resolution
	{"n_fft": 1024, "hop_length": 256, "n_mels": 512},  # Longer hop length
]

compare_spectrograms(audio_file, params_list)
