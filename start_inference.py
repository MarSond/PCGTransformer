import os
import os.path
from run import Run

run1_name = "run1"

if __name__ == "__main__":
	print("Start inference pipeline")

	run1 = Run(config_update_dict={"load_config_from_run_name": run1_name})