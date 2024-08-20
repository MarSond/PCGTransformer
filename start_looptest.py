from pathlib import Path

import matplotlib.pyplot as plt

from MLHelper.constants import *
from MLHelper.tools.utils import MLUtil
from run import Run


def send_result_mail(results: dict):
	subject = f"Training Complete: {results['run_name']}"
	body = f"Training for has completed successfully.\n\nResults:\n{results}"
	to_email = "martinsondermann10@gmail.com"  # Ihre E-Mail-Adresse
	from_email = "martinsondermann10@gmail.com"  # Ihre Gmail-Adresse

	with Path("email_password.txt").open() as f:
		password = f.read().strip()

	if MLUtil.send_email(subject, body, to_email, from_email, password):
		print("Email notification sent.")
	else:
		print("Failed to send email notification.")

def do_run(config: dict):
	run = Run(config_update_dict=config)
	run.setup_task()
	result = run.start_task()
	send_result_mail(result)
	print(result)


if __name__ == "__main__":

	train_update_dict = {	TASK_TYPE: TRAINING, METADATA_FRAC: 1.0, \
							CNN_PARAMS: {}, EPOCHS: 50, BATCH_SIZE: 80, \
							SINGLE_BATCH_MODE: False, TRAIN_FRAC: 0.8, KFOLD_SPLITS: 1, \
							# TRAINING_CHECKPOINT: {EPOCH: 70, RUN_NAME: "run1", FOLD: 6}, \
							CHUNK_DURATION: 7.0, CHUNK_METHOD: CHUNK_METHOD_FIXED, \
							DO_FAKE_UPDATES: 0, RUN_NAME_SUFFIX: "test", \
							MODEL_METHOD_TYPE: BEATS
                      }


	run_loop_test_dict = train_update_dict.copy()
	run_loop_test_dict.update({ \
		TRAIN_DATASET: PHYSIONET_2022, KFOLD_SPLITS: 3, EPOCHS: 2, METADATA_FRAC: 0.07})

	do_run(run_loop_test_dict)

	plt.show()

	# continue_test = {METADATA_FRAC: 0.05, TRAINING_CHECKPOINT: {EPOCH: 80, RUN_NAME: "2024-07-12_21-52-51_combined-optimized-2", FOLD: 1}, \
	#	LOAD_PREVIOUS_RUN_NAME: "2024-07-12_21-52-51_combined-optimized-2" ,RUN_NAME_SUFFIX: "continue-test", EPOCHS: 82}
