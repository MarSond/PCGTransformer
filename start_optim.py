import json

import optuna

from MLHelper.constants import *
from run import Run

# ruff: noqa: T201

def do_run(config: dict):
	try:
		run = Run(config_update_dict=config)
		run.setup_task()
		return run.start_task(), run.run_name
	except Exception as e:
		print(e)
		print("Failed to start training.")
		return None

def objective(trial):
	# Definiere die Hyperparameter, die Optuna optimieren soll
	train_update_dict = {
		TASK_TYPE: TRAINING,
		METADATA_FRAC: 0.03,
		CNN_PARAMS: {},
		EPOCHS: 2,
		TRAIN_FRAC: trial.suggest_float(TRAIN_FRAC, 0.7, 0.9),
		KFOLD_SPLITS: 1,
		CHUNK_DURATION: trial.suggest_int(CHUNK_DURATION, 3.0, 12.0),
		CHUNK_METHOD: CHUNK_METHOD_FIXED,
		# Weitere Parameter...
	}

	# Update für spezifische Runs
	train_update_dict.update({
		TRAIN_DATASET: PHYSIONET_2022,
		RUN_NAME_SUFFIX: "optuna_run"
	})

	# Starte den Trainingslauf
	result = do_run(train_update_dict)

	if result is not None:
		if METRICS_MCC in result:
			result_value = result[METRICS_MCC]
			trial.set_user_attr("run_name", result[RUN_NAME])
		else:
			result_value = - 1.0
	else:
		result_value = -1.0
	return result_value

if __name__ == "__main__":
	def trial_callback(study, trial):
		print(f"Trial {trial.number} finished with value: {trial.value} and params: {trial.params}")
		# Ergebnis in eine Datei schreiben
		with open("optuna_trials.log", "a") as f:
			f.write(json.dumps({"date_start": trial.date_start,
				"date_complete": trial.date_complete,
				"duration": trial.duration,
				"user_attrs": trial.user_attrs,
				"number": trial.number, "value": trial.value, "params": trial.params}, f)+"\n")

	# Erstelle eine Optuna-Studie
	study_name = "example-study"  # Unique identifier of the study.
	file_storage = optuna.storages.JournalStorage(
		optuna.storages.JournalFileStorage("./optuna_journal.log"),  # NFS path for distributed optimization
	)
	storage_name = "sqlite:///{}.db".format(study_name)

	study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)
	study.optimize(objective, n_trials=60, callbacks=[trial_callback])

	# Optuna-Ergebnisse anzeigen
	print("Best trial:")
	trial = study.best_trial

	print("Value: ", trial.value)
	print("Params: ")
	for key, value in trial.params.items():
		print(f"{key}: {value}")
