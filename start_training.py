import matplotlib.pyplot as plt

from MLHelper.constants import *
from run import Run


def do_run(config: dict):
	run = Run(config_update_dict=config)
	run.setup_task()
	run.start_task()


if __name__ == "__main__":
	"""
	TODO verify checkpointing works. Current run1 checkpoint has bad performance in training
	In standalone inference it works well.
	Thesis: it doesnt work well because LR scheduler and scaler,
	which are used in training werent saved in the model
	"""

	train_update_dict = {	TASK_TYPE: TRAINING, METADATA_FRAC: 0.8, \
							CNN_PARAMS: {}, EPOCHS: 50, \
							SINGLE_BATCH_MODE: False, TRAIN_FRAC: 0.8, KFOLD_SPLITS: 1, \
							# TRAINING_CHECKPOINT: {EPOCH: 70, RUN_NAME: "run1", FOLD: 6}, \
							CHUNK_DURATION: 7.0, CHUNK_METHOD: CHUNK_METHOD_FIXED, \
                      }

	run1_dict = train_update_dict.copy()
	run1_dict.update({TRAIN_DATASET: PHYSIONET_2022, RUN_NAME_SUFFIX: "run2022"})

	run2_dict = train_update_dict.copy()
	run2_dict.update({TRAIN_DATASET: PHYSIONET_2016, RUN_NAME_SUFFIX: "run2016"})

	run3_dict = train_update_dict.copy()
	run3_dict.update({TRAIN_DATASET: PHYSIONET_2016_2022, RUN_NAME_SUFFIX: "run2022-2016"})

	run4_dict = train_update_dict.copy()
	run4_dict.update({TRAIN_DATASET: PHYSIONET_2022, CHUNK_METHOD: CHUNK_METHOD_CYCLES, CHUNK_DURATION: 7.0, \
		CHUNK_HEARTCYCLE_COUNT: 10, AUDIO_LENGTH_NORM: LENGTH_NORM_STRETCH, RUN_NAME_SUFFIX: "cycles-10"})

	run5_dict = train_update_dict.copy()
	run5_dict.update({TRAIN_DATASET: PHYSIONET_2022, CHUNK_METHOD: CHUNK_METHOD_CYCLES, CHUNK_DURATION: 5.0, \
		CHUNK_HEARTCYCLE_COUNT: 5, AUDIO_LENGTH_NORM: LENGTH_NORM_STRETCH, RUN_NAME_SUFFIX: "cycles-5"})

	# Experiment 6: Adjust mel spectrogram parameters
	run6_dict = run1_dict.copy()
	run6_dict.update({
		RUN_NAME_SUFFIX: "mel_adjust-half",
		CNN_PARAMS: {
			N_MELS: 256,
			HOP_LENGTH: 64,
			N_FFT: 512,
		}
	})

	# Experiment 7: Modify signal filtering
	run7_dict = run1_dict.copy()
	run7_dict.update({
		RUN_NAME_SUFFIX: "differentsignal_filter",
		CNN_PARAMS: {
			SIGNAL_FILTER: BUTTERPASS,
			BUTTERPASS_LOW: 20,
			BUTTERPASS_HIGH: 800,
			BUTTERPASS_ORDER: 3,
		}
	})

	# Experiment 8: Adjust learning rate and optimizer
	run8_dict = run1_dict.copy()
	run8_dict.update({
		RUN_NAME_SUFFIX: "lr_optimizer",
		LEARNING_RATE: 0.001,
		CNN_PARAMS: {
			OPTIMIZER: OPTIMIZER_ADAM,
			SCHEDULER: SCHEDULER_COSINE,
		}
	})

	# Experiment 9: Modify model architecture
	run9_dict = run1_dict.copy()
	run9_dict.update({
		RUN_NAME_SUFFIX: "dropouthigher",
		CNN_PARAMS: {
			DROP0: 0.7,
			DROP1: 0.4,
		}
	})

	run10_dict = run1_dict.copy()
	run10_dict.update({
		RUN_NAME_SUFFIX: "mel_adjust-double",
		CNN_PARAMS: {
			N_MELS: 1024,
			HOP_LENGTH: 256,
			N_FFT: 2048,
			}
	})


	# Experiment 11: long chunks
	run11_dict = run1_dict.copy()
	run11_dict.update({
		RUN_NAME_SUFFIX: "long-chunks",
		CHUNK_DURATION: 15.0
	})

	# Experiment 12: Regularization
	run12_dict = run1_dict.copy()
	run12_dict.update({
		RUN_NAME_SUFFIX: "regularization",
		CNN_PARAMS: {
			L1_REGULATION_WEIGHT: 0.0001,
			L2_REGULATION_WEIGHT: 0.0001,
		},
		TRAINING_CHECKPOINT: {EPOCH: 41, RUN_NAME: "2024-07-10_17-14-30_regularization", FOLD: 1},
	})

	# Experiment 13: No filter
	run13_dict = run1_dict.copy()
	run13_dict.update({
		RUN_NAME_SUFFIX: "nofilter",
		CNN_PARAMS: {
			SIGNAL_FILTER: None,
		}
	})


	# Experiment 14: short chunks
	run14_dict = run1_dict.copy()
	run14_dict.update({
		RUN_NAME_SUFFIX: "short-chunks",
		CHUNK_DURATION: 4.0
	})


	# Experiment 15: 
	run15_dict = run1_dict.copy()
	run15_dict.update({
		RUN_NAME_SUFFIX: "mel-clear",
		CNN_PARAMS: {
			N_MELS: 512,
			HOP_LENGTH: 64,
			N_FFT: 256,
		}
	})


	# Experiment 16: 
	run16_dict = run1_dict.copy()
	run16_dict.update({
		RUN_NAME_SUFFIX: "model2",
		CNN_PARAMS: {
			MODEL_SUB_TYPE: 2,
		}
	})


	# Experiment 17:
	run17_dict = run1_dict.copy()
	run17_dict.update({
		RUN_NAME_SUFFIX: "model3",
		CNN_PARAMS: {
			MODEL_SUB_TYPE: 3,
		}
	})

	# Experiment 18:
	run18_dict = run1_dict.copy()
	run18_dict.update({
		RUN_NAME_SUFFIX: "long-epoch-adam",
		EPOCHS: 100,
		LEARNING_RATE: 0.01,
		CNN_PARAMS: {
			OPTIMIZER: OPTIMIZER_ADAM,
			SCHEDULER: SCHEDULER_PLATEAU,
		}
	})

	# Experiment 19:
	run19_dict = run1_dict.copy()
	run19_dict.update({
		RUN_NAME_SUFFIX: "model4",
		CNN_PARAMS: {
			MODEL_SUB_TYPE: 4,
		}
	})

	###

	run_loop_test_dict = train_update_dict.copy()
	run_loop_test_dict.update({ \
		TRAIN_DATASET: PHYSIONET_2022, KFOLD_SPLITS: 3, EPOCHS: 4, METADATA_FRAC: 0.05})

	# do_run(run1_dict)
	# do_run(run2_dict)
	# do_run(run3_dict)
	# do_run(run4_dict)
	# do_run(run5_dict)

	# do_run(run6_dict)
	# do_run(run7_dict)
	# do_run(run8_dict)
	# do_run(run9_dict)
	# do_run(run10_dict)

	# do_run(run11_dict)
	# do_run(run12_dict)
	# do_run(run13_dict)
	# do_run(run14_dict)
	# do_run(run15_dict)
	# do_run(run16_dict)
	# do_run(run17_dict)
	do_run(run18_dict)
	do_run(run19_dict)

	# TODO test no downsampling of 2022
	#do_run(run_loop_test_dict)

	plt.show()
