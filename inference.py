from run import Run
from MLHelper.config import Config

# Path to folder to a saved run inside test_models
test_model_name = ""

def prepare_run() -> Run:
	run = Run()
	
	return run

if __name__ == "__main__":
	prepare_run()