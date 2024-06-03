from MLHelper.constants import *
from run import Run

run1_name = "run1"

# Attention: using LOAD_PREVIOUS_RUN_NAME loads the config saved by that specific run. 
# Extra config from the current update_dict is applied afterwards. Keep in mind!

if __name__ == "__main__":

	inference_update_dict = {	TASK_TYPE: INFERENCE, METADATA_FRAC: 0.2, \
									INFERENCE_MODEL: {EPOCHS: 70, FOLD: 10}, \
								BATCH_SIZE: 1, EPOCHS: 1,  \
								SINGLE_BATCH_MODE: False, TRAIN_FRAC: 0.0, \
								INFERENCE_DATASET: PHYSIONET_2016, \
								CHUNK_DURATION: 7.0, CHUNK_METHOD: CHUNK_METHOD_FIXED, \
							}

	run1_dict = inference_update_dict.copy()
	run1_dict.update({LOAD_PREVIOUS_RUN_NAME: run1_name})
	run1 = Run(config_update_dict=run1_dict)
	run1.setup_task()
	run1.start_task()

