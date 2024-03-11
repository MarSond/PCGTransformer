import MLHelper.constants as const
from run import Run

run1_name = "run1"

if __name__ == "__main__":
	print("Start inference pipeline")
	inference_update_dict = { "task_type": "inference", const.METADATA_FRAC: 0.1, const.INFERENCE_MODEL: {const.EPOCHS: 70, const.KFOLD: 10} }

	run1_dict = inference_update_dict.copy()
	run1_dict.update({const.LOAD_PREVIOUS_RUN_NAME: run1_name})
	run1 = Run(config_update_dict=run1_dict)

	run1.start_task()