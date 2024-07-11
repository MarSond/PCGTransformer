import matplotlib.pyplot as plt

from MLHelper.constants import *
from run import Run


def do_run(config: dict):
	run = Run(config_update_dict=config)
	run.setup_task()
	run.start_task()


if __name__ == "__main__":

	train_update_dict = {	TASK_TYPE: TRAINING, METADATA_FRAC: 0.8, \
							CNN_PARAMS: {}, EPOCHS: 50, \
							SINGLE_BATCH_MODE: False, TRAIN_FRAC: 0.8, KFOLD_SPLITS: 1, \
							# TRAINING_CHECKPOINT: {EPOCH: 70, RUN_NAME: "run1", FOLD: 6}, \
							CHUNK_DURATION: 7.0, CHUNK_METHOD: CHUNK_METHOD_FIXED, \
							DO_FAKE_UPDATES: 0, RUN_NAME_SUFFIX: "test", \
                      }


	run_loop_test_dict = train_update_dict.copy()
	run_loop_test_dict.update({ \
		TRAIN_DATASET: PHYSIONET_2022, KFOLD_SPLITS: 3, EPOCHS: 4, METADATA_FRAC: 0.05})

	do_run(run_loop_test_dict)

	plt.show()
