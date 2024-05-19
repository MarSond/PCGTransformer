from MLHelper.constants import *
from run import Run


if __name__ == "__main__":

	train_update_dict = {	TASK_TYPE: TRAINING, METADATA_FRAC: 0.2, \
							CNN_PARAMS: { },  \
							SINGLE_BATCH_MODE: False, TRAIN_FRAC: 0.8, \
							TRAIN_DATASET: PHYSIONET_2022, KFOLD_SPLITS: 0, \
							#TRAINING_CHECKPOINT: {EPOCH: 10, RUN_NAME: "run1_name"}, \
							CHUNK_DURATION: 7.0, CHUNK_METHOD: CHUNK_METHOD_FIXED, \
						}

	run1_dict = train_update_dict.copy()

	run1 = Run(config_update_dict=run1_dict)
	run1.setup_task()
	run1.start_task()

