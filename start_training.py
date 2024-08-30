from pathlib import Path

import matplotlib.pyplot as plt

from MLHelper.constants import *
from MLHelper.tools.utils import MLUtil
from run import Run

# ruff: noqa: T201

def send_result_mail(name: str, results: dict):
	try:
		subject = f"Study Complete: {name}"
		body = f"Study completed successfully.\n\nStudy name: {name}\nresults:\n{results}"
		own_adress = "martinsondermann10@gmail.com"

		if MLUtil.send_self_mail_gmail(subject, body, own_adress):
			print("Email notification sent.")
		else:
			print("Failed to send email notification.")
	except Exception as e:
		print(f"Failed to send email notification: {e}")

def do_run(config: dict):
	try:
		run = Run(config_update_dict=config)
		run.setup_task()
		result = run.start_task()
		send_result_mail(config[RUN_NAME_SUFFIX], result)
	except Exception as e:
		print(e)
		print("Failed to start training.")


if __name__ == "__main__":

	base_config = {	TASK_TYPE: TRAINING, METADATA_FRAC: 1.0, \
					SINGLE_BATCH_MODE: False, KFOLD_SPLITS: 10, SIGNAL_FILTER: BUTTERPASS, \
					SAVE_ONLY_LAST_MODEL: False, SAVE_MODEL: True, \
					EARLY_STOPPING_ENABLED: False,
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
		EPOCHS: 60,
		MODEL_METHOD_TYPE: CNN,
		CHUNK_METHOD: CHUNK_METHOD_FIXED,
		RUN_NAME_SUFFIX: "2016_fixed_cnn_fullrun-final3",

		CHUNK_DURATION: 10.0,
		NORMALIZATION: NORMALIZATION_ZSCORE,
		AUDIO_LENGTH_NORM: LENGTH_NORM_STRETCH,
		OPTIMIZER: OPTIMIZER_ADAMW,
		L1_REGULATION_WEIGHT: 1.0e-05,
		L2_REGULATION_WEIGHT: 0.0005,
		LEARNING_RATE: 0.0001,
		AUGMENTATION_RATE: 0.6,
		SCHEDULER: SCHEDULER_PLATEAU,
		SCHEDULER_FACTOR: 0.5,
		SCHEDULER_PATIENCE: 10,
		CNN_PARAMS: {
			ACTIVATION: ACTIVATION_SILU,
			DROP0: 0.3,
			DROP1: 0.6,
			N_MELS: 512,
			HOP_LENGTH: 128,
			N_FFT: 512,
			MODEL_SUB_TYPE: 2,
		}
	})

	physionet_2016_fixed_beats = base_config.copy()
	physionet_2016_fixed_beats.update({
		TRAIN_DATASET: PHYSIONET_2016,
		EPOCHS: 50,
		KFOLD_SPLITS: 5,
		MODEL_METHOD_TYPE: BEATS,
		CHUNK_METHOD: CHUNK_METHOD_FIXED,
		RUN_NAME_SUFFIX: "2016_fixed_beats_fullrun-final2",

		CHUNK_DURATION: 10.0,
		NORMALIZATION: NORMALIZATION_ZSCORE,
		AUDIO_LENGTH_NORM: LENGTH_NORM_STRETCH,
		OPTIMIZER: OPTIMIZER_ADAMW,
		L1_REGULATION_WEIGHT:  1.0e-05,
		L2_REGULATION_WEIGHT: 0.0005,
		LEARNING_RATE: 0.0001,
		AUGMENTATION_RATE: 0.6,
		SCHEDULER: SCHEDULER_PLATEAU,
		SCHEDULER_FACTOR: 0.5,
		SCHEDULER_PATIENCE: 10,
		TRANSFORMER_PARAMS: {
			ACTIVATION: ACTIVATION_SILU,
			DROP0: 0.3,
			DROP1: 0.6,
			MODEL_SUB_TYPE: 2,
			N_FFT: 512,
			HOP_LENGTH: 128,
			N_MELS: 512,
		}
	})

	# PhysioNet 2022 configurations
	physionet_2022_fixed_cnn = base_config.copy()
	physionet_2022_fixed_cnn.update({
		TRAIN_DATASET: PHYSIONET_2022,
		EPOCHS: 60,
		KFOLD_SPLITS: 5,
		MODEL_METHOD_TYPE: CNN,
		CHUNK_METHOD: CHUNK_METHOD_FIXED,
		RUN_NAME_SUFFIX: "2022_fixed_cnn_fullrun-final2",

		CHUNK_DURATION: 6.0,
		NORMALIZATION: NORMALIZATION_MAX_ABS,
		AUDIO_LENGTH_NORM: LENGTH_NORM_REPEAT,
		OPTIMIZER: OPTIMIZER_ADAM,
		L1_REGULATION_WEIGHT: 1.0e-05,
		L2_REGULATION_WEIGHT: 0.001,
		LEARNING_RATE: 6.0e-05,
		AUGMENTATION_RATE: 0.6,
		SCHEDULER: SCHEDULER_PLATEAU,
		SCHEDULER_FACTOR: 0.5,
		SCHEDULER_PATIENCE: 10,
		CNN_PARAMS: {
			ACTIVATION: ACTIVATION_SILU,
			DROP0: 0.3,
			DROP1: 0.6,
			N_MELS: 1024,
			HOP_LENGTH: 160,
			N_FFT: 640,
			MODEL_SUB_TYPE: 4, # TODO: Frage, wie weit sollten sich zum vergleich die modelle Unterscheiden? Vergleichbarkeit
		}
	})

	physionet_2022_cycles_cnn = base_config.copy()
	physionet_2022_cycles_cnn.update({
		TRAIN_DATASET: PHYSIONET_2022,
		EPOCHS: 60,
		MODEL_METHOD_TYPE: CNN,
		CHUNK_METHOD: CHUNK_METHOD_CYCLES,
		RUN_NAME_SUFFIX: "2022_cycles_cnn_fullrun-final2",

		CHUNK_DURATION: 10.0,
		CHUNK_HEARTCYCLE_COUNT: 6,
		NORMALIZATION: NORMALIZATION_ZSCORE,
		AUDIO_LENGTH_NORM: LENGTH_NORM_STRETCH,
		OPTIMIZER: OPTIMIZER_ADAMW,
		L1_REGULATION_WEIGHT: 0.00014,
		L2_REGULATION_WEIGHT: 0.00026,
		LEARNING_RATE: 2.0e-05,
		AUGMENTATION_RATE: 0.6,
		SCHEDULER: SCHEDULER_PLATEAU,
		SCHEDULER_FACTOR: 0.5,
		SCHEDULER_PATIENCE: 10,
		CNN_PARAMS: {
			ACTIVATION: ACTIVATION_RELU,
			DROP0: 0.3,
			DROP1: 0.6,
			N_MELS: 640,
			HOP_LENGTH: 224,
			N_FFT: 1408,
			MODEL_SUB_TYPE: 3, # TODO: Frage, wie weit sollten sich zum vergleich die modelle Unterscheiden? Vergleichbarkeit
		}
	})

	physionet_2022_fixed_beats = base_config.copy()
	physionet_2022_fixed_beats.update({
		TRAIN_DATASET: PHYSIONET_2022,
		EPOCHS: 50,
		KFOLD_SPLITS: 5,
		BATCH_SIZE: 5,
		MODEL_METHOD_TYPE: BEATS,
		GRAD_ACCUMULATE_STEPS: 7,
		CHUNK_METHOD: CHUNK_METHOD_FIXED,
		RUN_NAME_SUFFIX: "2022_fixed_beats_fullrun-final2",

		CHUNK_DURATION: 12.0,
		NORMALIZATION: NORMALIZATION_MAX_ABS,
		AUDIO_LENGTH_NORM: LENGTH_NORM_PADDING,
		OPTIMIZER: OPTIMIZER_ADAM,
		L1_REGULATION_WEIGHT: 0.005,
		L2_REGULATION_WEIGHT: 0.001,
		LEARNING_RATE: 0.0005,
		AUGMENTATION_RATE: 0.6,
		SCHEDULER: SCHEDULER_PLATEAU,
		SCHEDULER_FACTOR: 0.25,
		SCHEDULER_PATIENCE: 12,
		TRANSFORMER_PARAMS: {
			ACTIVATION: ACTIVATION_SILU,
			DROP0: 0.4,
			DROP1: 0.6,
			MODEL_SUB_TYPE: 3,
		}
	})

	physionet_2022_cycles_beats = base_config.copy()
	physionet_2022_cycles_beats.update({
		TRAIN_DATASET: PHYSIONET_2022,
		EPOCHS: 50,
		KFOLD_SPLITS: 5,
		BATCH_SIZE: 5,
		MODEL_METHOD_TYPE: BEATS,
		GRAD_ACCUMULATE_STEPS: 7,
		CHUNK_METHOD: CHUNK_METHOD_CYCLES,
		RUN_NAME_SUFFIX: "2022_cycle_beats_fullrun-final2",

		CHUNK_DURATION: 8.0,
		CHUNK_HEARTCYCLE_COUNT: 10,
		NORMALIZATION: NORMALIZATION_MAX_ABS,
		AUDIO_LENGTH_NORM: LENGTH_NORM_STRETCH,
		OPTIMIZER: OPTIMIZER_ADAMW,
		L1_REGULATION_WEIGHT: 0.0017,
		L2_REGULATION_WEIGHT: 1e-05,
		LEARNING_RATE: 4.1e-05,
		AUGMENTATION_RATE: 0.6,
		SCHEDULER: SCHEDULER_PLATEAU,
		SCHEDULER_FACTOR: 0.5,
		SCHEDULER_PATIENCE: 10,
		TRANSFORMER_PARAMS: {
			ACTIVATION: ACTIVATION_RELU,
			DROP0: 0.6,
			DROP1: 0.6,
			MODEL_SUB_TYPE: 2,
		}
	})

	# Execute runs
	do_run(physionet_2016_fixed_cnn) # done nmcc 0.859	# 3: 
	#do_run(physionet_2022_fixed_cnn) # done nmcc 0.76	# 3:
	#do_run(physionet_2022_cycles_beats) # done bad		# 3:

	#do_run(physionet_2022_fixed_beats)
	#do_run(physionet_2022_cycles_cnn)

	#do_run(physionet_2016_fixed_beats)

	plt.show()
