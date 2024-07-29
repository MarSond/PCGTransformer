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




	beats_2022_5s_m2 = train_update_dict.copy()
	beats_2022_5s_m2.update({
		TRAIN_DATASET: PHYSIONET_2022,
		MODEL_SUB_TYPE:2,
		MODEL_METHOD_TYPE: BEATS,
		BATCH_SIZE: 16,
		CHUNK_DURATION: 5.0,
		RUN_NAME_SUFFIX: "beats-2022-5s-model2"
	})
	#do_run(beats_2022_5s_m2)

	plt.show()