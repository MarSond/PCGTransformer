import json
import logging
from pathlib import Path

import optuna

from MLHelper.constants import *
from run import Run

# ruff: noqa: T201

def do_run(config: dict):
	try:
		run = Run(config_update_dict=config)
		run.setup_task()
		return run.start_task()
	except Exception as e:
		logging.error(f"Failed to start training: {e}")
		return None

def get_common_update_dict(trial, model_type):
	return {
		TASK_TYPE: TRAINING,
		METADATA_FRAC: 0.8,
		TRAIN_FRAC: 0.8,
		KFOLD_SPLITS: 1,
		RUN_NAME_SUFFIX: f"optuna_{PHYSIONET_2022}_{model_type.lower()}_{trial.number}",

		LEARNING_RATE: trial.suggest_float(f"{model_type.lower()}_lr", 1e-5, 1e-2, log=True),
		L1_REGULATION_WEIGHT: trial.suggest_float(L1_REGULATION_WEIGHT, 1e-7, 1e-2, log=True),
		L2_REGULATION_WEIGHT: trial.suggest_float(L2_REGULATION_WEIGHT, 1e-7, 1e-2, log=True),
		OPTIMIZER: trial.suggest_categorical(OPTIMIZER, [OPTIMIZER_ADAM, OPTIMIZER_SGD, OPTIMIZER_ADAMW]),
		SCHEDULER: trial.suggest_categorical(SCHEDULER, [SCHEDULER_COSINE, SCHEDULER_STEP, SCHEDULER_PLATEAU]),
		SIGNAL_FILTER: trial.suggest_categorical(SIGNAL_FILTER, [BUTTERPASS, None]),
		# AUGMENTATION_RATE: trial.suggest_float(AUGMENTATION_RATE, 0.0, 1.0, step=0.1),
	}

def get_beats_update_dict(trial):
	ud = get_common_update_dict(trial, BEATS)
	ud.update({
		EPOCHS: 15,
		BATCH_SIZE: 16,
		MODEL_METHOD_TYPE: BEATS,
		TRANSFORMER_PARAMS: {
			DROP0: trial.suggest_float("beats_drop0", 0.2, 0.8, step=0.2),
			DROP1: trial.suggest_float("beats_drop1", 0.2, 0.6, step=0.1),
			ACTIVATION: trial.suggest_categorical(ACTIVATION, [ACTIVATION_SILU, ACTIVATION_RELU]),
			MODEL_SUB_TYPE: trial.suggest_int(MODEL_SUB_TYPE, 1, 3),
		},
	})
	return ud

def get_cnn_update_dict(trial):
	ud = get_common_update_dict(trial, CNN)
	ud.update({
		EPOCHS: 25,
		BATCH_SIZE: 80,
		MODEL_METHOD_TYPE: CNN,
		NORMALIZATION: trial.suggest_categorical(NORMALIZATION, [NORMALIZATION_MAX_ABS, NORMALIZATION_ZSCORE]),
		CNN_PARAMS: {
			DROP0: trial.suggest_float(DROP0, 0.2, 0.8, step=0.2),
			DROP1: trial.suggest_float(DROP1, 0.2, 0.6, step=0.1),
			ACTIVATION: trial.suggest_categorical(ACTIVATION, [ACTIVATION_SILU, ACTIVATION_RELU]),
			MODEL_SUB_TYPE: trial.suggest_int(MODEL_SUB_TYPE, 1, 4),
			N_MELS: trial.suggest_int(N_MELS, 128, 1024, step=128),
			HOP_LENGTH: trial.suggest_int(HOP_LENGTH, 64, 256, step=32),
			N_FFT: trial.suggest_int(N_FFT, 128, 1024, step=128),
		},
	})
	return ud

def set_chunk_and_scheduler_params(ud, trial, dataset, chunk_method):
	ud[TRAIN_DATASET] =	dataset,
	ud[CHUNK_METHOD] = chunk_method
	if ud[CHUNK_METHOD] == CHUNK_METHOD_CYCLES:
		ud[CHUNK_HEARTCYCLE_COUNT] = trial.suggest_int(CHUNK_HEARTCYCLE_COUNT, 3, 15, step=1)
		ud[AUDIO_LENGTH_NORM] = LENGTH_NORM_STRETCH
	else:
		ud[AUDIO_LENGTH_NORM] = trial.suggest_categorical( \
			AUDIO_LENGTH_NORM, [LENGTH_NORM_PADDING, LENGTH_NORM_REPEAT, LENGTH_NORM_STRETCH])

	if ud[SCHEDULER] in [SCHEDULER_PLATEAU, SCHEDULER_STEP, SCHEDULER_COSINE]:
		ud[SCHEDULER_PATIENCE] = trial.suggest_int(SCHEDULER_PATIENCE, 5, 15, step=1)
		ud[SCHEDULER_FACTOR] = trial.suggest_float(SCHEDULER_FACTOR, 0.1, 0.9, step=0.1)

def objective(trial, get_update_dict, dataset, chunk_method):
	train_update_dict = get_update_dict(trial)
	set_chunk_and_scheduler_params(train_update_dict, trial, dataset, chunk_method)
	result = do_run(train_update_dict)

	if result is not None and METRICS_NMCC in result:
		trial.set_user_attr("run_name", result[RUN_NAME])
		return result[METRICS_MCC]
	return 0.0

def trial_callback(study, trial):
	print(f"Trial {trial.number} finished with value: {trial.value}")
	print(f"Parameters: {json.dumps(trial.params, indent=2)}")

	trial_data = {
		"date_start": trial.datetime_start.isoformat(),
		"date_complete": trial.datetime_complete.isoformat(),
		"duration": str(trial.duration),
		"user_attrs": trial.user_attrs,
		"number": trial.number,
		"value": trial.value,
		"params": trial.params
	}


	with Path(f"{study.study_name}_trials.log").open("a") as f:
		for key, value in trial_data.items():
			f.write(f"{key}: {value}\n")
		f.write("\n#\n")

def start_optimization(model_type, n_trials, dataset, chunk_method):

	study_name = f"{model_type.lower()}_{dataset}_{chunk_method}"
	storage_name = f"sqlite:///optim_{dataset}.db"

	study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True, direction="maximize")
	def objective_func(trial):
		return objective(trial, get_beats_update_dict if model_type == BEATS else get_cnn_update_dict)
	study.optimize(objective_func, n_trials=n_trials, callbacks=[trial_callback])

	print("Best trial:")
	trial = study.best_trial
	print(f"Value: {trial.value}")
	print("Params:")
	for key, value in trial.params.items():
		print(f"  {key}: {value}")

if __name__ == "__main__":
	start_optimization(BEATS, n_trials=20, dataset=PHYSIONET_2022, chunk_method=CHUNK_METHOD_CYCLES)
	start_optimization(CNN, n_trials=20, dataset=PHYSIONET_2022, chunk_method=CHUNK_METHOD_CYCLES)
	start_optimization(BEATS, n_trials=20, dataset=PHYSIONET_2022, chunk_method=CHUNK_METHOD_FIXED)
	start_optimization(CNN, n_trials=20, dataset=PHYSIONET_2022, chunk_method=CHUNK_METHOD_FIXED)

	start_optimization(BEATS, n_trials=20, dataset=PHYSIONET_2016, chunk_method=CHUNK_METHOD_FIXED)
	start_optimization(CNN, n_trials=20, dataset=PHYSIONET_2016, chunk_method=CHUNK_METHOD_FIXED)
