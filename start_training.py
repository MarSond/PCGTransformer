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
	Thesis: it doesnt work well because LR sheduler and scaler,
	which are used in training werent saved in the model
	"""

	train_update_dict = {	TASK_TYPE: TRAINING, METADATA_FRAC: 0.8, \
							CNN_PARAMS: {}, EPOCHS: 72, \
							SINGLE_BATCH_MODE: False, TRAIN_FRAC: 0.8, KFOLD_SPLITS: 10, \
							TRAINING_CHECKPOINT: {EPOCH: 70, RUN_NAME: "run1", FOLD: 6}, \
							CHUNK_DURATION: 7.0, CHUNK_METHOD: CHUNK_METHOD_FIXED, \
                      }

	run1_dict = train_update_dict.copy()
	run1_dict.update({TRAIN_DATASET: PHYSIONET_2022, RUN_NAME_SUFFIX: "run2022"})

	run2_dict = train_update_dict.copy()
	run2_dict.update({TRAIN_DATASET: PHYSIONET_2016, RUN_NAME_SUFFIX: "run2016"})

	run3_dict = train_update_dict.copy()
	run3_dict.update({TRAIN_DATASET: PHYSIONET_2022, RUN_NAME_SUFFIX: "run2022", CHUNK_DURATION: 4.0})

	run4_dict = train_update_dict.copy()
	run4_dict.update({TRAIN_DATASET: PHYSIONET_2022, RUN_NAME_SUFFIX: "run2022-nofilter", \
						CNN_PARAMS: {SIGNAL_FILTER: None}})

	run_loop_test_dict = train_update_dict.copy()
	#run_loop_test_dict.update({ \
	#	TRAIN_DATASET: PHYSIONET_2022, KFOLD_SPLITS: 3, EPOCHS: 4, METADATA_FRAC: 0.05})

	#do_run(run1_dict)
	#do_run(run2_dict)
	#do_run(run3_dict)
	#do_run(run4_dict)

	do_run(run_loop_test_dict)
	plt.show()
