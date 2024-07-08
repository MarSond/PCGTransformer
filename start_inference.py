import matplotlib.pyplot as plt

from MLHelper.constants import *
from run import Run

run1_name = "run1"

# Attention: using LOAD_PREVIOUS_RUN_NAME loads the config saved by that specific run.
# Extra config from the current update_dict is applied afterwards. Keep in mind!

def do_run(config: dict):
	run = Run(config_update_dict=config)
	run.setup_task()
	run.start_task()

if __name__ == "__main__":
	inference_base = {TASK_TYPE: INFERENCE, BATCH_SIZE: 1, EPOCHS: 1, \
		TRAIN_FRAC: 0.0, L1_REGULATION_WEIGHT: 0.0, L2_REGULATION_WEIGHT: 0.0, DO_FAKE_UPDATES: 0}

	inference_update_dict_1 = {	METADATA_FRAC: 1.0, \
								INFERENCE_MODEL: {EPOCHS: 70, FOLD: 10}, \
								L1_REGULATION_WEIGHT: 0.0, L2_REGULATION_WEIGHT: 0.0, \
								CHUNK_DURATION: 7.0, CHUNK_METHOD: CHUNK_METHOD_FIXED, \
							}
	inference_update_dict_1.update(inference_base)

	run1_dict = inference_update_dict_1.copy()
	run1_dict.update({LOAD_PREVIOUS_RUN_NAME: run1_name, INFERENCE_DATASET: PHYSIONET_2016})

	run2_dict = inference_update_dict_1.copy()
	run2_dict.update({LOAD_PREVIOUS_RUN_NAME: run1_name, INFERENCE_DATASET: PHYSIONET_2022})


	#do_run(run1_dict)
	do_run(run2_dict)

	plt.show()

