import matplotlib.pyplot as plt

from MLHelper.constants import *
from run import Run

# ruff: noqa: T201

def do_run(config: dict):
	try:
		run = Run(config_update_dict=config)
		run.setup_task()
		run.start_task()
	except Exception as e:
		print(e)
		print("Failed to start training.")


if __name__ == "__main__":

	base_config = {	TASK_TYPE: TRAINING, METADATA_FRAC: 1.0, \
					SINGLE_BATCH_MODE: False, TRAIN_FRAC: 0.8, KFOLD_SPLITS: 1, SIGNAL_FILTER: BUTTERPASS, \
							# TRAINING_CHECKPOINT: {EPOCH: 70, RUN_NAME: "run1", FOLD: 6},
				}

	cycle_base = base_config.copy()
	cycle_base.update({TRAIN_DATASET: PHYSIONET_2022, CHUNK_METHOD: CHUNK_METHOD_CYCLES, \
		AUDIO_LENGTH_NORM: LENGTH_NORM_STRETCH, CHUNK_HEARTCYCLE_COUNT: 5})

	run_2022 = base_config.copy()
	run_2022.update({TRAIN_DATASET: PHYSIONET_2022, RUN_NAME_SUFFIX: "run2022"})
	#do_run(run_2022)

	run_2016 = base_config.copy()
	run_2016.update({TRAIN_DATASET: PHYSIONET_2016, RUN_NAME_SUFFIX: "run2016"})
	#do_run(run_2016)

	# Multi axies HPO: LR+L1+L2+OPTIMIZER ; SCHEDULER+FACTOR+PATIENCE ; nftt+hopl+nmels ; model+drop0+drop1

	# PhysioNet 2016 configurations
	physionet_2016_fixed_cnn = base_config.copy()
	physionet_2016_fixed_cnn.update({
		TRAIN_DATASET: PHYSIONET_2016,
		EPOCHS: 80,
		KFOLD_SPLITS: 10,
		MODEL_METHOD_TYPE: CNN,
		CHUNK_METHOD: CHUNK_METHOD_FIXED,
		RUN_NAME_SUFFIX: "2016_fixed_cnn_fullrun",

		NORMALIZATION: NORMALIZATION_MAX_ABS,
		AUDIO_LENGTH_NORM: LENGTH_NORM_PADDING,
		OPTIMIZER: OPTIMIZER_ADAMW,
		L1_REGULATION_WEIGHT: 6.0e-06,
		L2_REGULATION_WEIGHT: 6.8e-06,
		LEARNING_RATE: 0.001,
		SCHEDULER: SCHEDULER_PLATEAU,
		SCHEDULER_FACTOR: 0.6,
		SCHEDULER_PATIENCE: 10,
		CNN_PARAMS: {
			ACTIVATION: ACTIVATION_RELU,
			DROP0: 0.2,
			DROP1: 0.6,
			N_MELS: 128,
			HOP_LENGTH: 256,
			N_FFT: 1024,
			MODEL_SUB_TYPE: 3,
		}
	})

	physionet_2016_fixed_beats = base_config.copy()
	physionet_2016_fixed_beats.update({
		TRAIN_DATASET: PHYSIONET_2016,
		CHUNK_METHOD: CHUNK_METHOD_FIXED,
		MODEL_METHOD_TYPE: BEATS,
		RUN_NAME_SUFFIX: "2016_fixed_beats"
	})

	# PhysioNet 2022 configurations
	physionet_2022_fixed_cnn = base_config.copy()
	physionet_2022_fixed_cnn.update({
		TRAIN_DATASET: PHYSIONET_2022,
		MODEL_METHOD_TYPE: CNN,
		CHUNK_METHOD: CHUNK_METHOD_FIXED,
		RUN_NAME_SUFFIX: "2022_fixed_cnn"
	})

	physionet_2022_cycles_cnn = base_config.copy()
	physionet_2022_cycles_cnn.update({
		TRAIN_DATASET: PHYSIONET_2022,
		CHUNK_METHOD: CHUNK_METHOD_CYCLES,
		AUDIO_LENGTH_NORM: LENGTH_NORM_STRETCH,
		MODEL_METHOD_TYPE: CNN,
		CHUNK_HEARTCYCLE_COUNT: 5,
		RUN_NAME_SUFFIX: "2022_cycles_cnn"
	})

	physionet_2022_fixed_beats = base_config.copy()
	physionet_2022_fixed_beats.update({
		TRAIN_DATASET: PHYSIONET_2022,
		CHUNK_METHOD: CHUNK_METHOD_FIXED,
		MODEL_METHOD_TYPE: BEATS,
		RUN_NAME_SUFFIX: "2022_fixed_beats"
	})

	physionet_2022_cycles_beats = base_config.copy()
	physionet_2022_cycles_beats.update({
		TRAIN_DATASET: PHYSIONET_2022,
		CHUNK_METHOD: CHUNK_METHOD_CYCLES,
		AUDIO_LENGTH_NORM: LENGTH_NORM_STRETCH,
		CHUNK_HEARTCYCLE_COUNT: 5,
		MODEL_METHOD_TYPE: BEATS,
		RUN_NAME_SUFFIX: "2022_cycles_beats"
	})

	# Execute runs
	do_run(physionet_2016_fixed_cnn)
	# do_run(physionet_2016_fixed_beats)
	# do_run(physionet_2022_fixed_cnn)
	# do_run(physionet_2022_cycles_cnn)
	# do_run(physionet_2022_fixed_beats)
	# do_run(physionet_2022_cycles_beats)

	plt.show()


	#do_run(beats_2022_5s_m2)

	plt.show()
