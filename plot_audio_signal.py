"""
Task is to create an overview of all audio files in the dataset.
"""
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from torch import Tensor

import MLHelper.constants as const
from cnn_classifier.cnn_dataset import CNN_Dataset
from MLHelper.audio import preprocessing
from MLHelper.audio.audioutils import AudioUtil
from run import Run


class DataAnalysis:
	def __init__(self, dataset_name: str):
		self.demo_task = const.TASK_TYPE_INFERENCE # with augmentation or not
		self.run = Run()
		self.run.config[const.TASK_TYPE] = const.TASK_TYPE_DEMO
		self.run.config[const.INFERENCE_DATASET] = dataset_name
		self.run.config[const.KFOLD_SPLITS] = 1
		self.run.config[const.METADATA_FRAC] = 1.0
		self.run.config[const.BATCH_SIZE] = 1
		self.run.config[const.AUGMENTATION_RATE] = 0.0

		"""
		self.run.config[const.AUDIO_LENGTH_NORM] = const.LENGTH_NORM_STRETCH
		self.run.config[const.CHUNK_METHOD] = const.CHUNK_METHOD_CYCLES
		self.run.config[const.CHUNK_PADDING_THRESHOLD] = 0.0
		self.run.config[const.CHUNK_DURATION] = 12
		self.run.config[const.CHUNK_HEARTCYCLE_COUNT] = 6
		"""

		self.run.config[const.AUDIO_LENGTH_NORM] = const.LENGTH_NORM_PADDING
		self.run.config[const.CHUNK_METHOD] = const.CHUNK_METHOD_CYCLES
		self.run.config[const.CHUNK_PADDING_THRESHOLD] = 0.65
		self.run.config[const.CHUNK_DURATION] = 15
		self.run.config[const.CHUNK_HEARTCYCLE_COUNT] = 6

		if self.demo_task == const.TASK_TYPE_TRAINING:
			self.run.config[const.TRAIN_FRAC] = 0.99
		else:
			self.run.config[const.TRAIN_FRAC] = 0.0

		self.run.setup_task()
		self.run.task.dataset.load_file_list()
		self.run.task.dataset.prepare_chunks()
		self.run.task.dataset.prepare_kfold_splits()
		train_loader, valid_loader = self.get_dataloaders()
		self.demo_loader = train_loader if self.demo_task == const.TASK_TYPE_TRAINING else valid_loader

	def get_dataloaders(self):
		train_loader, valid_loader, _ = \
			self.run.task.dataset.get_dataloaders(num_split=1, dataset_class=CNN_Dataset)
		return train_loader, valid_loader

	def make_singlefile_plot(self, raw_audio, filtered_audio, full_audio, sgram_raw, sgram_filtered, \
							sgram_augmented, meta_row, audio_file_name, ax=None, final_only=False):
		class_id = meta_row[const.META_LABEL_1]
		# get sr from file meta_row[const.META_SAMPELRATE] and compare to sr target
		sr_factor = self.run.task.dataset.target_samplerate / meta_row[const.META_SAMPLERATE]
		sr = meta_row[const.META_SAMPLERATE] * sr_factor
		cycle_marker = meta_row[const.META_HEARTCYCLES]
		#cycle_marker = [marker * sr_factor for marker in cycle_marker]

		cnn_config = self.run.config[const.CNN_PARAMS]
		config = self.run.config
		frame_start = meta_row[const.CHUNK_RANGE_START]
		frame_end = meta_row[const.CHUNK_RANGE_END]
		hop_length = cnn_config[const.HOP_LENGTH]

		# Berechnen der relativen Position der cycle_marker im Ausschnitt
		relative_cycle_markers = [marker for marker in cycle_marker \
			if frame_start <= marker*meta_row[const.META_SAMPLERATE] <= frame_end]
		#remove markers outside of the frame
		audio_length = len(full_audio)
		relative_cycle_markers = [marker for marker in relative_cycle_markers \
				if 0 <= marker*meta_row[const.META_SAMPLERATE] <= audio_length]
		relative_cycle_markers = \
			[marker - frame_start/meta_row[const.META_SAMPLERATE] for marker in relative_cycle_markers]

		if final_only:
			if ax is not None:
				ax3, ax6, ax_text = ax
			else:
				# only plot text, full audio and final mel
				_ = plt.figure(figsize=(15, 10))
				# Die vierte Zeile ist 0.2 mal so hoch wie die anderen
				gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 0.2])
				ax3 = plt.subplot(gs[0, 0]) # full audio
				ax6 = plt.subplot(gs[1, 0])
				ax_text = plt.subplot(gs[2, :])
		else:
				gs = gridspec.GridSpec(4, 2, height_ratios=[1, 1, 1, 0.2])
				ax1 = plt.subplot(gs[0, 0]) # raw
				ax2 = plt.subplot(gs[0, 1]) # filtered
				ax3 = plt.subplot(gs[1, 0]) # full audio
				ax4 = plt.subplot(gs[1, 1]) # raw mel
				ax5 = plt.subplot(gs[2, 0]) # filtered mel
				ax6 = plt.subplot(gs[2, 1]) # augmented mel
				ax_text = plt.subplot(gs[3, :])

		if class_id == 1:
			ax3.set_title(f"Full Audio Class ID: {class_id}", color="red")
		elif class_id == 0:
			ax3.set_title(f"Full Audio Class ID: {class_id}", color="green")
		else:
			ax3.set_title(f"Raw Audio Class ID: {class_id}")

		if not final_only:


			# Erster Subplot für das rohe Audiosignal
			AudioUtil.SignalPlotting.show_signal( \
				raw_audio, samplerate=sr, raw=True, ax=ax1, cycle_marker=relative_cycle_markers)
			# red if class is 1, green on class 0

			# Zweiter Subplot für das bearbeitete Audiosignal
			AudioUtil.SignalPlotting.show_signal(filtered_audio, samplerate=sr, \
				raw=True, ax=ax2, cycle_marker=relative_cycle_markers)
			ax2.set_title("Filtered Audio")

			# Vierter Subplot für das rohe Mel-Spektrogramm
			AudioUtil.SignalPlotting.show_mel_spectrogram(sgram_raw, samplerate=sr, ax=ax4, hop_length=hop_length)
			ax4.set_title("Raw Mel Spectrogram")


			# Fünfter Subplot für das bearbeitete Mel-Spektrogramm
			AudioUtil.SignalPlotting.show_mel_spectrogram(sgram_filtered, samplerate=sr, ax=ax5, hop_length=hop_length)
			ax5.set_title("Filtered Mel Spectrogram")

		# Dritter Subplot für das komplette Audiosignal
		AudioUtil.SignalPlotting.show_signal( \
			full_audio, samplerate=sr, raw=True, ax=ax3, cycle_marker=meta_row[const.META_HEARTCYCLES])
		ax3.set_title("Full length Audio")
		# markieren der aktuellen Position im full audio
		ax3.axvspan(frame_start / meta_row[const.META_SAMPLERATE], \
			frame_end / meta_row[const.META_SAMPLERATE], color="red", alpha=0.05)


		# Sechster Subplot für das augmentierte Mel-Spektrogramm
		AudioUtil.SignalPlotting.show_mel_spectrogram(sgram_augmented, samplerate=sr, ax=ax6, hop_length=hop_length)
		ax6.set_title("Augmented final Mel Spectrogram")

		# Letzter, flacher Subplot für den Text  # Nimmt beide Spalten ein
		ax_text.axis("off")  # Keine Achsen für diesen Subplot
		text_content = f"Raw Audio: {raw_audio.min():.4f} - {raw_audio.max():.4f}\n" \
						f"Processed Audio: {filtered_audio.min():.4f} - {filtered_audio.max():.4f}\n" \
						f"Class ID: {class_id} | samplerate: {sr} | " \
						f"chunk seconds: {config[const.CHUNK_DURATION]}\n" \
						f"Butterworth: {config[const.BUTTERPASS_LOW]} - {config[const.BUTTERPASS_HIGH]} " \
						f"- {config[const.BUTTERPASS_ORDER]}\n" \
						f"Mel Spectrogramm: {sgram_raw.min():.4f} - {sgram_raw.max():.4f}\n" \
						f"Filename: {audio_file_name}"
		ax_text.text(0.5, 0.5, text_content, ha="center", va="center", fontsize=11, wrap=True)

		if final_only:
			return ax3, ax6, ax_text
		return ax1, ax2, ax3, ax4, ax5, ax6, ax_text

	def plot_signal_stats(self, num_samples=10, offset=0, show=True, fig_file_name=None, final_only=False):
		offset += 1 # offset starts with 1, depends on order in the loop and how skip is used
		# loop train and valid loader back to back
		loader_counter = 0
		if num_samples == 0:
			num_samples = len(self.demo_loader)
		if num_samples + offset > len(self.demo_loader):
			num_samples = len(self.demo_loader) - offset
		num_gs = 3 if final_only else 7
		fig = plt.figure(figsize=(5*num_gs, 4*num_samples))
		gs = gridspec.GridSpec(num_samples, num_gs)  # 10 Reihen für die Samples, 5 Spalten für die Subplots
		for raw_audio, filtered_audio, sgram_raw, sgram_filtered, sgram_augmented, metadata_row, audio_file_name in self.demo_loader:
			audio_file_name = str(audio_file_name[0])
			audio_path = Path(self.run.task.dataset.dataset_path) / metadata_row[const.META_AUDIO_PATH][0]
			full_audio, full_sr, padding = AudioUtil.Loading.load_audiofile(audio_path)
			full_audio = preprocessing.resample(full_audio, full_sr, self.run.task.dataset.target_samplerate)
			# check values in the dict metadata_row to see if they are tensors- > convert to numpy
			if isinstance(metadata_row[const.META_SAMPLERATE], Tensor):
				metadata_row[const.META_SAMPLERATE] = metadata_row[const.META_SAMPLERATE].numpy().item()
				metadata_row[const.META_LABEL_1] = metadata_row[const.META_LABEL_1].numpy().item()
				metadata_row[const.META_HEARTCYCLES] = \
					[ x.numpy().item() for x in metadata_row[const.META_HEARTCYCLES]]
				metadata_row[const.CHUNK_RANGE_START] = metadata_row[const.CHUNK_RANGE_START].numpy().item()
				metadata_row[const.CHUNK_RANGE_END] = metadata_row[const.CHUNK_RANGE_END].numpy().item()
			if isinstance(raw_audio, Tensor):
				raw_audio = raw_audio.numpy()
				filtered_audio = filtered_audio.numpy()
				sgram_raw = sgram_raw.numpy()
				sgram_filtered = sgram_filtered.numpy()
				sgram_augmented = sgram_augmented.numpy()
			if full_audio.ndim > 1:
				full_audio = full_audio.squeeze()
			loader_counter += 1
			if loader_counter < offset:
				continue
			if loader_counter > num_samples + offset-1:
				break
			print(f"Audio filename: {audio_file_name} - Counter: - {loader_counter}")
			row_index = loader_counter - offset - 1  # Zeilenindex für die Unterfiguren

			if not final_only:
				ax1 = plt.subplot(gs[row_index, 0])
				ax2 = plt.subplot(gs[row_index, 1])
				ax3 = plt.subplot(gs[row_index, 2])
				ax4 = plt.subplot(gs[row_index, 3])
				ax5 = plt.subplot(gs[row_index, 4])
				ax6 = plt.subplot(gs[row_index, 5])
				ax_text = plt.subplot(gs[row_index, 6])
				try:
					self.make_singlefile_plot(raw_audio, filtered_audio, full_audio, sgram_raw, \
								sgram_filtered, sgram_augmented, metadata_row, audio_file_name, \
								ax=(ax1, ax2, ax3, ax4, ax5, ax6, ax_text), final_only=final_only)
				except Exception as e:
					print(f"Error in display_sample {e}")
					# fill ax blank
					ax1.axis("off")
					ax2.axis("off")
					ax3.axis("off")
					ax4.axis("off")
					ax5.axis("off")
					ax6.axis("off")
					ax_text.axis("off")
					continue
			else:
				ax3 = plt.subplot(gs[row_index, 0])
				ax6 = plt.subplot(gs[row_index, 1])
				ax_text = plt.subplot(gs[row_index, 2])
				ax3, ax6, ax_text = self.make_singlefile_plot(raw_audio, filtered_audio, full_audio, \
					sgram_raw, sgram_filtered, sgram_augmented, metadata_row, audio_file_name, \
					ax=(ax3,ax6,ax_text),final_only=final_only)

			#########
		#plt.tight_layout()

		if fig_file_name:
			fig_file_name = f"audio_example_images/{fig_file_name}_{num_samples}_samples_{offset}_offset.png"
			fig.savefig(fig_file_name)
		if show:
			plt.show()

def multiple():
	analysis = DataAnalysis(const.PHYSIONET_2022)
	length = len(analysis.demo_loader)
	max_per_run = 70
	for i in range(0, length, max_per_run):
		analysis.plot_signal_stats(num_samples=max_per_run, offset=i, show=False, \
			fig_file_name="ph2022", final_only=True)
		# release memory
		plt.close("all")
	# analysis2 = DataAnalysis(const.PHYSIONET_2016)
	# length = len(analysis2.demo_loader)
	# max_per_run = 50
	# for i in range(0, length, max_per_run):
	# 	analysis2.plot_signal_statistics(num_samples=max_per_run, offset=i, show=False, fig_file_name="ph2016")
	# 	plt.close("all")


def single():
	analysis = DataAnalysis(const.PHYSIONET_2016)
	analysis.plot_signal_stats(num_samples=5, offset=19, show=True, fig_file_name=None, final_only=True)


if __name__ == "__main__":
	#multiple()
	single()
