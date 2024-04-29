import MLHelper.constants as const
from run import Run

run1_name = "run1"

if __name__ == "__main__":
	print("Start inference pipeline")
	inference_update_dict = { 	const.TASK_TYPE: const.INFERENCE, const.METADATA_FRAC: 1.0, \
						  		const.INFERENCE_MODEL: {const.EPOCHS: 70, const.KFOLD: 10}, \
								const.CNN_PARAMS: {const.BATCH_SIZE: 1},  \
								const.INFERENCE_DATASET: const.PHYSIONET_2016, \
								const.SINGLE_BATCH_MODE: False, const.TRAIN_FRAC: 0.0, \
							}

	run1_dict = inference_update_dict.copy()
	run1_dict.update({const.LOAD_PREVIOUS_RUN_NAME: run1_name})
	run1 = Run(config_update_dict=run1_dict)
	run1.setup_task()
	run1.start_task()

