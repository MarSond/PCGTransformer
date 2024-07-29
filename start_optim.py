import json

import optuna

from MLHelper.constants import *
from run import Run

# ruff: noqa: T201, SIM114

def do_run(config: dict):
	try:
		run = Run(config_update_dict=config)
		run.setup_task()
		return run.start_task(), run.run_name
	except Exception as e:
		print(e)
		print("Failed to start training.")
		return None

def get_beats_update_dict(trial):
	ud = {
		TASK_TYPE: TRAINING,
		METADATA_FRAC: 1.0,
		EPOCHS: 20,
		TRAIN_FRAC: 0.8,
		KFOLD_SPLITS: 1,
		BATCH_SIZE: 16,
		MODEL_METHOD_TYPE: BEATS,
		TRAIN_DATASET: PHYSIONET_2022,
		RUN_NAME_SUFFIX: "optuna_run_beats_" + str(trial.number),

		CHUNK_DURATION: trial.suggest_int(CHUNK_DURATION, 3, 18, step=2),
		CHUNK_METHOD: trial.suggest_categorical(CHUNK_METHOD, [CHUNK_METHOD_FIXED, CHUNK_METHOD_CYCLES]),

		LEARNING_RATE: trial.suggest_float("beats_lr", 1e-5, 1e-2, log=True),
		TRANSFORMER_PARAMS: {
			DROP0: trial.suggest_float("beats_drop0", 0.2, 0.8, step=0.2),
			DROP1: trial.suggest_float("beats_drop1", 0.2, 0.6, step=0.1),
			ACTIVATION: trial.suggest_categorical(ACTIVATION, [ACTIVATION_SILU, ACTIVATION_RELU]),
			MODEL_SUB_TYPE: trial.suggest_int(MODEL_SUB_TYPE, 1, 3),
		},

		L1_REGULATION_WEIGHT: trial.suggest_float(L1_REGULATION_WEIGHT, 1e-7, 1e-2, log=True),
		L2_REGULATION_WEIGHT: trial.suggest_float(L2_REGULATION_WEIGHT, 1e-7, 1e-2, log=True),
		OPTIMIZER: trial.suggest_categorical(OPTIMIZER, [OPTIMIZER_ADAM, OPTIMIZER_SGD, OPTIMIZER_ADAMW]),
		SCHEDULER: trial.suggest_categorical(SCHEDULER, [SCHEDULER_COSINE, SCHEDULER_STEP, SCHEDULER_PLATEAU, None]),
		SIGNAL_FILTER: trial.suggest_categorical(SIGNAL_FILTER, [BUTTERPASS, None]),
		AUDIO_LENGTH_NORM: LENGTH_NORM_REPEAT,
		CHUNK_HEARTCYCLE_COUNT: 5,
		SCHEDULER_PATIENCE: 10,
		SCHEDULER_FACTOR: 0.1,
	}

	if ud[CHUNK_METHOD] == CHUNK_METHOD_CYCLES:
		ud[CHUNK_HEARTCYCLE_COUNT] = trial.suggest_int(CHUNK_HEARTCYCLE_COUNT, 3, 15, step=1)
		ud[AUDIO_LENGTH_NORM] = LENGTH_NORM_STRETCH

	if ud[CHUNK_METHOD] == CHUNK_METHOD_FIXED:
		ud[AUDIO_LENGTH_NORM] = trial.suggest_categorical(AUDIO_LENGTH_NORM, [LENGTH_NORM_PADDING, LENGTH_NORM_REPEAT]),

	if ud[SCHEDULER] == SCHEDULER_PLATEAU:
		ud[SCHEDULER_PATIENCE] = trial.suggest_int(SCHEDULER_PATIENCE, 5, 15, step=1)
		ud[SCHEDULER_FACTOR] = trial.suggest_float(SCHEDULER_FACTOR, 0.1, 0.9, step=0.1)
	elif ud[SCHEDULER] == SCHEDULER_STEP:
		ud[SCHEDULER_PATIENCE] = trial.suggest_int(SCHEDULER_PATIENCE, 5, 15, step=1)
		ud[SCHEDULER_FACTOR] = trial.suggest_float(SCHEDULER_FACTOR, 0.1, 0.9, step=0.1)
	elif ud[SCHEDULER] == SCHEDULER_COSINE:
		ud[SCHEDULER_PATIENCE] = trial.suggest_int(SCHEDULER_PATIENCE, 5, 15, step=1)
		ud[SCHEDULER_FACTOR] = trial.suggest_float(SCHEDULER_FACTOR, 0.1, 0.9, step=0.1)

	return ud

def get_cnn_update_dict(trial):
	ud = {
		TASK_TYPE: TRAINING,
		METADATA_FRAC: 1.0,
		EPOCHS: 25,
		TRAIN_FRAC: 0.8,
		KFOLD_SPLITS: 1,
		BATCH_SIZE: 86,
		MODEL_METHOD_TYPE: CNN,
		TRAIN_DATASET: PHYSIONET_2022,
		RUN_NAME_SUFFIX: "optuna_run_cnn_" + str(trial.number),

		CHUNK_DURATION: trial.suggest_int(CHUNK_DURATION, 3, 15, step=2),
		CHUNK_METHOD: trial.suggest_categorical(CHUNK_METHOD, [CHUNK_METHOD_FIXED, CHUNK_METHOD_CYCLES]),

		LEARNING_RATE: trial.suggest_float(LEARNING_RATE, 1e-5, 1e-2, log=True),
		CNN_PARAMS: {
			DROP0: trial.suggest_float(DROP0, 0.2, 0.8, step=0.2),
			DROP1: trial.suggest_float(DROP1, 0.2, 0.6, step=0.1),
			ACTIVATION: trial.suggest_categorical(ACTIVATION, [ACTIVATION_SILU, ACTIVATION_RELU]),
			MODEL_SUB_TYPE: trial.suggest_int(MODEL_SUB_TYPE, 1, 4),

			N_MELS: trial.suggest_int(N_MELS, 128, 1024, step=128),
			HOP_LENGTH: trial.suggest_int(HOP_LENGTH, 64, 256, step=32),
			N_FFT: trial.suggest_int(N_FFT, 128, 1024, step=128),
		},

		L1_REGULATION_WEIGHT: trial.suggest_float(L1_REGULATION_WEIGHT, 1e-7, 1e-2, log=True),
		L2_REGULATION_WEIGHT: trial.suggest_float(L2_REGULATION_WEIGHT, 1e-7, 1e-2, log=True),
		OPTIMIZER: trial.suggest_categorical(OPTIMIZER, [OPTIMIZER_ADAM, OPTIMIZER_SGD, OPTIMIZER_ADAMW]),
		SCHEDULER: trial.suggest_categorical(SCHEDULER, [SCHEDULER_COSINE, SCHEDULER_STEP, SCHEDULER_PLATEAU, None]),
		SIGNAL_FILTER: trial.suggest_categorical(SIGNAL_FILTER, [BUTTERPASS, None]),
		AUDIO_LENGTH_NORM: LENGTH_NORM_REPEAT,
		CHUNK_HEARTCYCLE_COUNT: 5,
		SCHEDULER_PATIENCE: 10,
		SCHEDULER_FACTOR: 0.1,
	}

	if ud[CHUNK_METHOD] == CHUNK_METHOD_CYCLES:
		ud[CHUNK_HEARTCYCLE_COUNT] = trial.suggest_int(CHUNK_HEARTCYCLE_COUNT, 3, 15, step=1)
		ud[AUDIO_LENGTH_NORM] = LENGTH_NORM_STRETCH

	if ud[CHUNK_METHOD] == CHUNK_METHOD_FIXED:
		ud[AUDIO_LENGTH_NORM] = trial.suggest_categorical(AUDIO_LENGTH_NORM, [LENGTH_NORM_PADDING, LENGTH_NORM_REPEAT]),

	if ud[SCHEDULER] == SCHEDULER_PLATEAU:
		ud[SCHEDULER_PATIENCE] = trial.suggest_int(SCHEDULER_PATIENCE, 5, 15, step=1)
		ud[SCHEDULER_FACTOR] = trial.suggest_float(SCHEDULER_FACTOR, 0.1, 0.9, step=0.1)
	elif ud[SCHEDULER] == SCHEDULER_STEP:
		ud[SCHEDULER_PATIENCE] = trial.suggest_int(SCHEDULER_PATIENCE, 5, 15, step=1)
		ud[SCHEDULER_FACTOR] = trial.suggest_float(SCHEDULER_FACTOR, 0.1, 0.9, step=0.1)
	elif ud[SCHEDULER] == SCHEDULER_COSINE:
		ud[SCHEDULER_PATIENCE] = trial.suggest_int(SCHEDULER_PATIENCE, 5, 15, step=1)
		ud[SCHEDULER_FACTOR] = trial.suggest_float(SCHEDULER_FACTOR, 0.1, 0.9, step=0.1)

	return ud


def beats_objective(trial):
	# Definiere die Hyperparameter, die Optuna optimieren soll
	train_update_dict = get_beats_update_dict(trial)

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

def cnn_objective(trial):
	# Definiere die Hyperparameter, die Optuna optimieren soll
	train_update_dict = get_cnn_update_dict(trial)

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

def trial_callback(study, trial):
		print(f"Trial {trial.number} of {study.study_name} finished with value: {trial.value} and params: {trial.params}")
		# Ergebnis in eine Datei schreiben
		with open(f"{study.study_name}.log", "a") as f:
			f.write(json.dumps({"date_start": trial.datetime_start,
				"date_complete": trial.datetime_complete,
				"duration": trial.duration,
				"user_attrs": trial.user_attrs,
				"number": trial.number, "value": trial.value, "params": trial.params}, indent=4, sort_keys=True, default=str)+"\n,\n")

def start_beats():
	study_name = "beats-study"  # Unique identifier of the study.
	storage_name = "sqlite:///{}.db".format(study_name)


	study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)
	study.optimize(beats_objective, n_trials=40, callbacks=[trial_callback])

	# Optuna-Ergebnisse anzeigen
	print("Best trial:")
	trial = study.best_trial

	print("Value: ", trial.value)
	print("Params: ")
	for key, value in trial.params.items():
		print(f"{key}: {value}")

def start_cnn():
	study_name = "cnn-study"  # Unique identifier of the study.
	storage_name = "sqlite:///{}.db".format(study_name)


	study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)
	study.optimize(cnn_objective, n_trials=40, callbacks=[trial_callback])

	# Optuna-Ergebnisse anzeigen
	print("Best trial:")
	trial = study.best_trial

	print("Value: ", trial.value)
	print("Params: ")
	for key, value in trial.params.items():
		print(f"{key}: {value}")

if __name__ == "__main__":

	#start_beats()
	start_cnn()
