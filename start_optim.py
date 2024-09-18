import gc
import importlib
import json
import sys
from pathlib import Path

import optuna
import torch

from MLHelper.constants import *
from MLHelper.tools.utils import MLUtil
from run import Run


# ruff: noqa: T201
def reset_pytorch_state():
	gc.collect()

	# Reload key torch modules
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
		torch.cuda.reset_peak_memory_stats()
		# Erzwinge Garbage Collection
		gc.collect()

		# Setze alle CUDA-Geräte zurück
		if torch.cuda.is_available():
			for i in range(torch.cuda.device_count()):
				torch.cuda.set_device(i)
				torch.cuda.empty_cache()
				torch.cuda.reset_peak_memory_stats()

		# Versuche, relevante Module neu zu laden
		modules_to_reload = ['torch', 'torch.cuda', 'torchvision', 'torchaudio']
		for module_name in modules_to_reload:
			if module_name in sys.modules:
				try:
					importlib.reload(sys.modules[module_name])
				except Exception as e:
					print(f"Konnte Modul {module_name} nicht neu laden: {e}")

		# Versuche, CUDA neu zu initialisieren
		if torch.cuda.is_available():
			try:
				torch.cuda.init()
			except Exception as e:
				print(f"Konnte CUDA nicht neu initialisieren: {e}")
	print("CUDA and PyTorch reloaded")

def send_result_mail(name: str, results: dict):
	try:
		subject = f"Study Complete: {name}"
		body = f"Study completed successfully.\n\nStudy name: {name}\nresults:\n{results}"
		adress = "martinsondermann10@gmail.com"

		if MLUtil.send_self_mail_gmail(subject, body, adress):
			print("Email notification sent.")
		else:
			print("Failed to send email notification.")
	except Exception as e:
		print(f"Failed to send email notification: {e}")

def do_run(config: dict):
	try:
		run = Run(config_update_dict=config)
		run.setup_task()
		result = run.start_task()
		return result
	except Exception as e:
		print(f"Fehler in do_run: {e}")
		return None
	finally:
		reset_pytorch_state()


def get_common_update_dict(trial, model_type, dataset, chunk_method):
	return {
		TASK_TYPE: TRAINING,
		METADATA_FRAC: 0.7,
		TRAIN_FRAC: 0.8,
		KFOLD_SPLITS: 1,
		SAVE_MODEL: False,
		RUN_NAME_SUFFIX: f"optuna_{dataset}_{chunk_method}_{model_type.lower()}_{trial.number}",

		LEARNING_RATE: trial.suggest_float("lr", 0.000001, 0.01, log=True),
		L1_REGULATION_WEIGHT: trial.suggest_float(L1_REGULATION_WEIGHT, 1e-7, 1e-1, log=True),
		L2_REGULATION_WEIGHT: trial.suggest_float(L2_REGULATION_WEIGHT, 1e-7, 1e-1, log=True),

		SCHEDULER: SCHEDULER_PLATEAU,

	}

def get_beats_update_dict(trial, dataset, chunk_method):
	ud = get_common_update_dict(trial, BEATS, dataset, chunk_method)
	ud.update({
		EPOCHS: 25,
		BATCH_SIZE: 5,
		MODEL_METHOD_TYPE: BEATS,
		GRAD_ACCUMULATE_STEPS: 7,
		OPTIMIZER: OPTIMIZER_ADAMW,
		TRANSFORMER_PARAMS: {
			DROP0: 0.4,
			DROP1: 0.6,
			ACTIVATION: ACTIVATION_RELU,
			MODEL_SUB_TYPE: 3,
		},
	})
	return ud

def get_cnn_update_dict(trial, dataset, chunk_method):
	ud = get_common_update_dict(trial, CNN, dataset, chunk_method)
	ud.update({
		EPOCHS: 30,
		BATCH_SIZE: 72,
		MODEL_METHOD_TYPE: CNN,
		NORMALIZATION: NORMALIZATION_MAX_ABS,
		OPTIMIZER: trial.suggest_categorical(OPTIMIZER, [OPTIMIZER_ADAM, OPTIMIZER_ADAMW]),
		CNN_PARAMS: {
			DROP0: 0.4,
			DROP1: 0.6,
			ACTIVATION: trial.suggest_categorical(ACTIVATION, [ACTIVATION_SILU, ACTIVATION_RELU]),
			MODEL_SUB_TYPE: trial.suggest_int(MODEL_SUB_TYPE, 1, 4),
			N_MELS: trial.suggest_int(N_MELS, 128, 2048, step=256),
			HOP_LENGTH: trial.suggest_int(HOP_LENGTH, 32, 512, step=32),
			N_FFT: trial.suggest_int(N_FFT, 128, 2048, step=256),
		},
	})
	return ud

def set_chunk_and_scheduler_params(ud, trial, dataset, chunk_method):
	ud[TRAIN_DATASET] =	dataset
	ud[CHUNK_METHOD] = chunk_method
	if ud[CHUNK_METHOD] == CHUNK_METHOD_CYCLES:
		ud[CHUNK_HEARTCYCLE_COUNT] = trial.suggest_int(CHUNK_HEARTCYCLE_COUNT, 3, 17, step=1)
		ud[AUDIO_LENGTH_NORM] = LENGTH_NORM_STRETCH
		ud[CHUNK_DURATION] = trial.suggest_float("cycle_"+CHUNK_DURATION, 5.0, 18.0, step=1.0)
	else:
		ud[AUDIO_LENGTH_NORM] = LENGTH_NORM_PADDING
		ud[CHUNK_DURATION] = trial.suggest_float("fix_"+CHUNK_DURATION, 5.0, 18.0, step=1.0)

	if ud[SCHEDULER] == SCHEDULER_PLATEAU:
		ud[SCHEDULER_PATIENCE] = 10
	elif ud[SCHEDULER] == SCHEDULER_STEP:
		ud[SCHEDULER_PATIENCE] = 15
	ud[SCHEDULER_FACTOR] = 0.5

def objective(trial, get_update_dict, dataset, chunk_method):
	train_update_dict = get_update_dict(trial, dataset, chunk_method)
	set_chunk_and_scheduler_params(train_update_dict, trial, dataset, chunk_method)
	result = do_run(train_update_dict)

	if result is not None and METRICS_NMCC in result:
		trial.set_user_attr(RUN_NAME, result[RUN_NAME])
		return result[METRICS_NMCC]
	else:
		print("error in objective")
		print(result)
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

	with Path(f"{FOLDER_OPTIMIZATION}/{study.study_name}_trials.log").open("a") as f:
		for key, value in trial_data.items():
			f.write(f"{key}: {value}\n")
		f.write("\n#\n\n")

def start_optimization(model_type, n_trials, dataset, chunk_method):
	study_name = f"{model_type.lower()}_{dataset}_{chunk_method}_2"

	storage_name = f"sqlite:///{FOLDER_OPTIMIZATION}/optim_survey_4.db"

	try:
		study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True, direction="maximize")
		def objective_func(trial):
			return objective(trial, get_beats_update_dict if model_type == BEATS else get_cnn_update_dict, dataset, chunk_method)
		study.optimize(objective_func, n_trials=n_trials, callbacks=[trial_callback])

		print("Best trial:")
		trial = study.best_trial
		print(f"Value: {trial.value}")
		print("Params:")
		for key, value in trial.params.items():
			print(f"	{key}: {value}")
	except Exception as e:
		print(f"Exception in study: {e}")
	finally:
		try:
			best_val = study.best_value if study.best_value is not None else 0.0
			best_params = study.best_params if study.best_params is not None else {}
		except Exception as e:
			best_val = -1
			best_params = {}
		#send_result_mail(study_name, {"value": best_val, "params": best_params})
		print("Study done")

if __name__ == "__main__":
	#start_optimization(CNN, n_trials=1, dataset=PHYSIONET_2022, chunk_method=CHUNK_METHOD_CYCLES)
	#start_optimization(CNN, n_trials=1, dataset=PHYSIONET_2022, chunk_method=CHUNK_METHOD_FIXED)
	start_optimization(BEATS, n_trials=1, dataset=PHYSIONET_2022, chunk_method=CHUNK_METHOD_FIXED)

	#start_optimization(CNN, n_trials=25, dataset=PHYSIONET_2016, chunk_method=CHUNK_METHOD_FIXED)
	#start_optimization(BEATS, n_trials=1, dataset=PHYSIONET_2022, chunk_method=CHUNK_METHOD_CYCLES) # mehr machen

# TODO ausarbeiting: 50209_PV.wav - viele eindeutige Zyklen nciht labeleed
