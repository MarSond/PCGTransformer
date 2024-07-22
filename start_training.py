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

	train_update_dict = {	TASK_TYPE: TRAINING, METADATA_FRAC: 1.0, \
							CNN_PARAMS: {}, EPOCHS: 20, \
							SINGLE_BATCH_MODE: False, TRAIN_FRAC: 0.8, KFOLD_SPLITS: 1, \
							# TRAINING_CHECKPOINT: {EPOCH: 70, RUN_NAME: "run1", FOLD: 6}, \
							CHUNK_DURATION: 5.0, CHUNK_METHOD: CHUNK_METHOD_FIXED, \
                      }

	cycle_base = train_update_dict.copy()
	cycle_base.update({TRAIN_DATASET: PHYSIONET_2022, CHUNK_METHOD: CHUNK_METHOD_CYCLES, \
		AUDIO_LENGTH_NORM: LENGTH_NORM_STRETCH, CHUNK_HEARTCYCLE_COUNT: 5})

	run_2022 = train_update_dict.copy()
	run_2022.update({TRAIN_DATASET: PHYSIONET_2022, RUN_NAME_SUFFIX: "run2022"})
	#do_run(run_2022)

	run_2016 = train_update_dict.copy()
	run_2016.update({TRAIN_DATASET: PHYSIONET_2016, RUN_NAME_SUFFIX: "run2016"})
	#do_run(run_2016)

	run2016_2022 = train_update_dict.copy()
	run2016_2022.update({TRAIN_DATASET: PHYSIONET_2016_2022, RUN_NAME_SUFFIX: "run2022-2016"})
	#do_run(run2016_2022)
	###


	run_10c_7s = cycle_base.copy()
	run_10c_7s.update({CHUNK_DURATION: 7.0, CHUNK_HEARTCYCLE_COUNT: 10, RUN_NAME_SUFFIX: "cycles-10"})
	#do_run(run_10c_7s)

	run5_5c_5s = cycle_base.copy()
	run5_5c_5s.update({CHUNK_DURATION: 5.0, CHUNK_HEARTCYCLE_COUNT: 5, RUN_NAME_SUFFIX: "cycles-5"})
	#do_run(run5_5c_5s)


	###

	run_loop_test_dict = train_update_dict.copy()
	run_loop_test_dict.update({ MODEL_METHOD_TYPE: CNN ,\
		TRAIN_DATASET: PHYSIONET_2022, KFOLD_SPLITS: 1, EPOCHS: 2, METADATA_FRAC: 0.05})
	#do_run(run_loop_test_dict)


	continue_test = {METADATA_FRAC: 0.05, TRAINING_CHECKPOINT: {EPOCH: 80, RUN_NAME: "2024-07-12_21-52-51_combined-optimized-2", FOLD: 1}, \
		LOAD_PREVIOUS_RUN_NAME: "2024-07-12_21-52-51_combined-optimized-2" ,RUN_NAME_SUFFIX: "continue-test", EPOCHS: 82}


	# TODO test no downsampling of 2022

	beats_5c_5s = cycle_base.copy()
	beats_5c_5s.update({TRAIN_DATASET: PHYSIONET_2022, MODEL_METHOD_TYPE: BEATS , BATCH_SIZE: 24, METADATA_FRAC: 1.0, CHUNK_HEARTCYCLE_COUNT: 5, CHUNK_DURATION: 5.0, RUN_NAME_SUFFIX: "beats-5c-5s"})
	#do_run(beats_5c_5s)

	beats_5c_3s = cycle_base.copy()
	beats_5c_3s.update({TRAIN_DATASET: PHYSIONET_2022, MODEL_METHOD_TYPE: BEATS , BATCH_SIZE: 24, METADATA_FRAC: 1.0, CHUNK_HEARTCYCLE_COUNT: 5, CHUNK_DURATION: 3.0, RUN_NAME_SUFFIX: "beats-5c-3s"})
	#do_run(beats_5c_3s)

	beats_10c_5s = cycle_base.copy()
	beats_10c_5s.update({TRAIN_DATASET: PHYSIONET_2022, MODEL_METHOD_TYPE: BEATS , BATCH_SIZE: 24, METADATA_FRAC: 1.0, CHUNK_HEARTCYCLE_COUNT: 10, CHUNK_DURATION: 5.0, RUN_NAME_SUFFIX: "beats-10c-5s"})
	#do_run(beats_10c_5s)

	beats_2016_4s = train_update_dict.copy()
	beats_2016_4s.update({TRAIN_DATASET: PHYSIONET_2016, MODEL_METHOD_TYPE: BEATS , BATCH_SIZE: 24, CHUNK_DURATION: 4.0, RUN_NAME_SUFFIX: "beats-2016-4s"})
	#do_run(beats_2016_4s)

	beats_2016_12s = train_update_dict.copy()
	beats_2016_12s.update({TRAIN_DATASET: PHYSIONET_2016, MODEL_METHOD_TYPE: BEATS , BATCH_SIZE: 24, CHUNK_DURATION: 12.0, RUN_NAME_SUFFIX: "beats-2016-12s"})
	#do_run(beats_2016_12s)

	beats_2016_4s = train_update_dict.copy()
	beats_2016_4s.update({TRAIN_DATASET: PHYSIONET_2016, MODEL_METHOD_TYPE: BEATS , BATCH_SIZE: 24, CHUNK_DURATION: 4.0, RUN_NAME_SUFFIX: "beats-2016-4s"})
	#do_run(beats_2016_4s)

	beats_2022_12s = train_update_dict.copy()
	beats_2022_12s.update({TRAIN_DATASET: PHYSIONET_2022, MODEL_METHOD_TYPE: BEATS , BATCH_SIZE: 24, CHUNK_DURATION: 12.0, RUN_NAME_SUFFIX: "beats-2022-12s"})
	#do_run(beats_2022_12s)

	beats_2022_5s = train_update_dict.copy()
	beats_2022_5s.update({TRAIN_DATASET: PHYSIONET_2022, MODEL_METHOD_TYPE: BEATS , BATCH_SIZE: 24, CHUNK_DURATION: 5.0, RUN_NAME_SUFFIX: "beats-2022-5s"})
	#do_run(beats_2022_5s)

	###

	beats_2016_7s = train_update_dict.copy()
	beats_2016_7s.update({TRAIN_DATASET: PHYSIONET_2016, MODEL_METHOD_TYPE: BEATS , \
		BATCH_SIZE: 16, CHUNK_DURATION: 7.0, EPOCHS: 50,RUN_NAME_SUFFIX: "beats-2016-7s-w", \
		OPTIMIZER: OPTIMIZER_ADAMW, LEARNING_RATE: 0.001})
	do_run(beats_2016_7s)

	plt.show()
