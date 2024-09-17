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

# ruff: noqa: T201, E501

def reset_pytorch_state():
	gc.collect()
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
		torch.cuda.reset_peak_memory_stats()
		for i in range(torch.cuda.device_count()):
			torch.cuda.set_device(i)
			torch.cuda.empty_cache()
			torch.cuda.reset_peak_memory_stats()

		modules_to_reload = ["torch", "torchvision", "torchaudio"]
		for module_name in modules_to_reload:
			if module_name in sys.modules:
				try:
					importlib.reload(sys.modules[module_name])
				except Exception as e:
					print(f"Konnte Modul {module_name} nicht neu laden: {e}")

		try:
			torch.cuda.init()
		except Exception as e:
			print(f"Konnte CUDA nicht neu initialisieren: {e}")

	print("CUDA and PyTorch reloaded")

def send_result_mail(name: str, results: dict):
	try:
		subject = f"Study Complete: {name}"
		body = f"Study completed successfully.\n\nStudy name: {name}\nresults:\n{results}"
		address = "martinsondermann10@gmail.com"

		if MLUtil.send_self_mail_gmail(subject, body, address):
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

def get_base_config():
	return {
		TASK_TYPE: TRAINING,
		METADATA_FRAC: 0.85,
		TRAIN_FRAC: 0.8,
		KFOLD_SPLITS: 1,
		EPOCHS: 1,
		MODEL_METHOD_TYPE: BEATS,
		BATCH_SIZE: 16,
		OPTIMIZER: None,
		SCHEDULER: None,
		SAVE_MODEL: False,
	}

def get_beats_knn_params(trial):
	params = {
		TRANSFORMER_PARAMS: {
			MODEL_SUB_TYPE: MODEL_TYPE_KNN,
		},
		EMBEDDING_PARAMS: {
			EMBEDDING_CLASSIFIER: CLASSIFIER_KNN,
			KNN_N_NEIGHBORS: trial.suggest_int(KNN_N_NEIGHBORS, 1, 31, step=2),
			KNN_WEIGHT: trial.suggest_categorical(KNN_WEIGHT, [KNN_WEIGHT_UNIFORM, KNN_WEIGHT_DISTANCE]),
			KNN_METRIC: trial.suggest_categorical(KNN_METRIC, [KNN_METRIC_EUCLIDEAN, KNN_METRIC_MANHATTAN]),

			USE_SMOTE: trial.suggest_categorical(USE_SMOTE, [True, False], ),
			USE_UMAP: trial.suggest_categorical(USE_UMAP, [True, False]),
			USE_HDBSCAN: trial.suggest_categorical(USE_HDBSCAN, [True, False]),

		},
	}
	if params[EMBEDDING_PARAMS][USE_UMAP]:
		params[EMBEDDING_PARAMS].update({
			EMBEDDINGS_REDUCE_UMAP_N_COMPONENTS: trial.suggest_int(EMBEDDINGS_REDUCE_UMAP_N_COMPONENTS, 2, 64, log=True),
			EMBEDDINGS_REDUCE_UMAP_N_NEIGHBORS: trial.suggest_int(EMBEDDINGS_REDUCE_UMAP_N_NEIGHBORS, 2, 100, log=True),
			EMBEDDINGS_REDUCE_UMAP_MIN_DIST: trial.suggest_float(EMBEDDINGS_REDUCE_UMAP_MIN_DIST, 0.0, 0.99),
		})
	if params[EMBEDDING_PARAMS][USE_HDBSCAN]:
		params[EMBEDDING_PARAMS].update({
			HDBSCAN_PARAM_MIN_CLUSTER_SIZE: trial.suggest_int(HDBSCAN_PARAM_MIN_CLUSTER_SIZE, 2, 64, log=True),
			HDBSCAN_PARAM_MIN_SAMPLES: trial.suggest_int(HDBSCAN_PARAM_MIN_SAMPLES, 2, 64, log=True),
		})
	return params

def get_chunk_params(trial, chunk_method):
	params = {CHUNK_METHOD: chunk_method}
	if chunk_method == CHUNK_METHOD_CYCLES:
		params.update({
			AUDIO_LENGTH_NORM: LENGTH_NORM_STRETCH, # Important for cycles
			CHUNK_HEARTCYCLE_COUNT: trial.suggest_int(CHUNK_HEARTCYCLE_COUNT, 3, 16, step=1),
			CHUNK_DURATION: trial.suggest_float(f"cycle_{CHUNK_DURATION}", 3.0, 16.0, step=1.0)
		})
	else:
		params.update({
			AUDIO_LENGTH_NORM: trial.suggest_categorical(AUDIO_LENGTH_NORM, [LENGTH_NORM_PADDING, LENGTH_NORM_REPEAT, LENGTH_NORM_STRETCH]),
			CHUNK_DURATION: trial.suggest_float(f"fix_{CHUNK_DURATION}", 3.0, 11.0, step=2.0)
		})
	return params

def objective(trial, config):
	update_dict = get_base_config()
	update_dict.update(config)
	update_dict.update(get_beats_knn_params(trial))
	update_dict.update(get_chunk_params(trial, config[CHUNK_METHOD]))
	update_dict[RUN_NAME_SUFFIX] = f"optuna_{config[TRAIN_DATASET]}_{config[CHUNK_METHOD]}_beats_knn_{trial.number}"

	result = do_run(update_dict)

	if result is not None and METRICS_NMCC in result:
		trial.set_user_attr(RUN_NAME, result[RUN_NAME])
		return result[METRICS_NMCC]
	else:
		print("Error in objective")
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

def start_optimization(config, n_trials):
	study_name = f"beats_knn_{config[TRAIN_DATASET]}_{config[CHUNK_METHOD]}"
	storage_name = f"sqlite:///{FOLDER_OPTIMIZATION}/optim_survey_3.db"

	try:
		study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True, direction="maximize")
		study.optimize(lambda trial: objective(trial, config), n_trials=n_trials, callbacks=[trial_callback])

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

	start_optimization({TRAIN_DATASET: PHYSIONET_2022, CHUNK_METHOD: CHUNK_METHOD_FIXED}, n_trials=2)
	start_optimization({TRAIN_DATASET: PHYSIONET_2022, CHUNK_METHOD: CHUNK_METHOD_CYCLES}, n_trials=2)
	start_optimization({TRAIN_DATASET: PHYSIONET_2016, CHUNK_METHOD: CHUNK_METHOD_FIXED}, n_trials=2)
	# Results -> Full extraction with best settings -> optim hdb, knn


# TODO CNN 4x