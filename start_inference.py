import os.path
from run import Run

run1_name = "run1"

if __name__ == "__main__":
	print("Start inference pipeline")
	inference_update_dict = { "task_type": "inference", "inference_model": "epoch_70_fold_10"}

	run1_dict = inference_update_dict.copy()
	run1_dict.update({"load_config_from_run_name": run1_name})
	run1 = Run(config_update_dict=run1_dict)

	run1.start_task()