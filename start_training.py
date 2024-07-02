from MLHelper.constants import *
from run import Run
import matplotlib.pyplot as plt

def do_run(config: dict):
	run = Run(config_update_dict=config)
	run.setup_task()
	run.start_task()


if __name__ == "__main__":

	train_update_dict = {	TASK_TYPE: TRAINING, METADATA_FRAC: 0.8,
							CNN_PARAMS: {}, EPOCHS: 50,
							SINGLE_BATCH_MODE: False, TRAIN_FRAC: 0.8, KFOLD_SPLITS: 1, \
							# TRAINING_CHECKPOINT: {EPOCH: 10, RUN_NAME: "run1_name"}, \
							CHUNK_DURATION: 7.0, CHUNK_METHOD: CHUNK_METHOD_FIXED, \
                      }

	run1_dict = train_update_dict.copy()
	run1_dict.update({TRAIN_DATASET: PHYSIONET_2022, RUN_NAME_SUFFIX: "run2022"})

	run2_dict = train_update_dict.copy()
	run2_dict.update({TRAIN_DATASET: PHYSIONET_2016, RUN_NAME_SUFFIX: "run2016"})

	run3_dict = train_update_dict.copy()
	run3_dict.update({TRAIN_DATASET: PHYSIONET_2022, RUN_NAME_SUFFIX: "run2022", CHUNK_DURATION: 4.0})

	run4_dict = train_update_dict.copy()
	run4_dict.update({TRAIN_DATASET: PHYSIONET_2022, RUN_NAME_SUFFIX: "run2022-nofilter", CNN_PARAMS: {SIGNAL_FILTER: None}})
	do_run(run1_dict)
	do_run(run2_dict)
	#do_run(run3_dict)
	do_run(run4_dict)
	plt.show()