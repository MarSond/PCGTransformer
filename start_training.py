from MLHelper.constants import *
from run import Run
import matplotlib.pyplot as plt

def do_run(config: dict):
	run = Run(config_update_dict=config)
	run.setup_task()
	run.start_task()


if __name__ == "__main__":

	train_update_dict = {	TASK_TYPE: TRAINING, METADATA_FRAC: 0.5,
							CNN_PARAMS: {}, EPOCHS: 60,
							SINGLE_BATCH_MODE: False, TRAIN_FRAC: 0.8, KFOLD_SPLITS: 1, \
							# TRAINING_CHECKPOINT: {EPOCH: 10, RUN_NAME: "run1_name"}, \
							CHUNK_DURATION: 7.0, CHUNK_METHOD: CHUNK_METHOD_FIXED, \
                      }

	run1_dict = train_update_dict.copy()
	run1_dict.update({TRAIN_DATASET: PHYSIONET_2022})

	run2_dict = train_update_dict.copy()
	run2_dict.update({TRAIN_DATASET: PHYSIONET_2016})

	#do_run(run1_dict)
	do_run(run2_dict)

	plt.show()