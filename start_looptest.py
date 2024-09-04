import matplotlib.pyplot as plt

from MLHelper.constants import *
from run import Run


def do_run(config: dict):
	run = Run(config_update_dict=config)
	run.setup_task()
	result = run.start_task()
	print(result)


if __name__ == "__main__":

	train_update_dict = {	TASK_TYPE: TRAINING, METADATA_FRAC: 1.0, \
							CNN_PARAMS: {}, EPOCHS: 50, BATCH_SIZE: 80, \
							SINGLE_BATCH_MODE: False, TRAIN_FRAC: 0.8, KFOLD_SPLITS: 1, \
							# TRAINING_CHECKPOINT: {EPOCH: 70, RUN_NAME: "run1", FOLD: 6}, \
							CHUNK_DURATION: 7.0, CHUNK_METHOD: CHUNK_METHOD_FIXED, \
							DO_FAKE_UPDATES: 0, RUN_NAME_SUFFIX: "beats-test", 
							EARLY_STOPPING_ENABLED: False, \
							MODEL_METHOD_TYPE: CNN
                      }


	run_loop_test_dict = train_update_dict.copy()
	run_loop_test_dict.update({ \
		TRAIN_DATASET: PHYSIONET_2016, KFOLD_SPLITS: 1, EPOCHS: 1, METADATA_FRAC: 0.05, \
		TRANSFORMER_PARAMS: {MODEL_SUB_TYPE: MODEL_TYPE_KNN}, \
		KNN_PARAMS: {KNN_N_NEIGHBORS: 3,
			KNN_ASSUME_POSTIVE_P: 0.5,
			KNN_WEIGHT: KNN_WEIGHT_UNIFORM,
			KNN_METRIC: KNN_METRIC_CHEBYSHEV,
			KNN_COMBINE_METHOD: KNN_COMBINE_METHOD_PLAIN
		},
	})

	do_run(run_loop_test_dict)

	plt.show()

	# continue_test = {METADATA_FRAC: 0.05, TRAINING_CHECKPOINT: {EPOCH: 80, RUN_NAME: "2024-07-12_21-52-51_combined-optimized-2", FOLD: 1}, \
	#	LOAD_PREVIOUS_RUN_NAME: "2024-07-12_21-52-51_combined-optimized-2" ,RUN_NAME_SUFFIX: "continue-test", EPOCHS: 82}
