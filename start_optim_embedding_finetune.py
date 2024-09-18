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
		print("starting in 3 sec...")
		import time
		time.sleep(3)
		run = Run(config_update_dict=config)
		run.setup_task()
		result = run.start_task()
		return result
	except Exception as e:
		print(f"Fehler in do_run: {e}")
		return None
	finally:
		reset_pytorch_state()

def get_base_config(model_config):
	base_config = {
		TASK_TYPE: TRAINING,
		METADATA_FRAC: 1.0,
		TRAIN_FRAC: 0.9,
		KFOLD_SPLITS: 1,
		EPOCHS: 1,
		MODEL_METHOD_TYPE: BEATS,
		BATCH_SIZE: 16,
		OPTIMIZER: None,
		SCHEDULER: None,
		SAVE_MODEL: False,
	}
	base_config.update(model_config)
	return base_config

def get_beats_knn_params(trial, param_ranges):
	params = {
		TRANSFORMER_PARAMS: {
			MODEL_SUB_TYPE: MODEL_TYPE_KNN,
		},
		EMBEDDING_PARAMS: {
			EMBEDDING_CLASSIFIER: CLASSIFIER_KNN,
			KNN_N_NEIGHBORS: trial.suggest_int(KNN_N_NEIGHBORS, *param_ranges[KNN_N_NEIGHBORS]),
			KNN_WEIGHT: trial.suggest_categorical(KNN_WEIGHT, param_ranges[KNN_WEIGHT]),
			KNN_METRIC: trial.suggest_categorical(KNN_METRIC, param_ranges[KNN_METRIC]),
			USE_SMOTE: trial.suggest_categorical(USE_SMOTE, param_ranges[USE_SMOTE]),
			USE_UMAP: trial.suggest_categorical(USE_UMAP, param_ranges[USE_UMAP]),
			USE_HDBSCAN: trial.suggest_categorical(USE_HDBSCAN, param_ranges[USE_HDBSCAN]),
		},
	}
	if params[EMBEDDING_PARAMS][USE_UMAP]:
		params[EMBEDDING_PARAMS].update({
			EMBEDDINGS_REDUCE_UMAP_N_COMPONENTS: trial.suggest_int(EMBEDDINGS_REDUCE_UMAP_N_COMPONENTS, *param_ranges[EMBEDDINGS_REDUCE_UMAP_N_COMPONENTS]),
			EMBEDDINGS_REDUCE_UMAP_N_NEIGHBORS: trial.suggest_int(EMBEDDINGS_REDUCE_UMAP_N_NEIGHBORS, *param_ranges[EMBEDDINGS_REDUCE_UMAP_N_NEIGHBORS]),
			EMBEDDINGS_REDUCE_UMAP_MIN_DIST: trial.suggest_float(EMBEDDINGS_REDUCE_UMAP_MIN_DIST, *param_ranges[EMBEDDINGS_REDUCE_UMAP_MIN_DIST]),
		})
	if params[EMBEDDING_PARAMS][USE_HDBSCAN]:
		params[EMBEDDING_PARAMS].update({
			HDBSCAN_PARAM_MIN_CLUSTER_SIZE: trial.suggest_int(HDBSCAN_PARAM_MIN_CLUSTER_SIZE, *param_ranges[HDBSCAN_PARAM_MIN_CLUSTER_SIZE]),
			HDBSCAN_PARAM_MIN_SAMPLES: trial.suggest_int(HDBSCAN_PARAM_MIN_SAMPLES, *param_ranges[HDBSCAN_PARAM_MIN_SAMPLES]),
		})
	return params

def get_chunk_params(trial, chunk_method, param_ranges):
	params = {CHUNK_METHOD: chunk_method}
	if chunk_method == CHUNK_METHOD_CYCLES:
		params.update({
			AUDIO_LENGTH_NORM: LENGTH_NORM_STRETCH,
			CHUNK_HEARTCYCLE_COUNT: trial.suggest_int(CHUNK_HEARTCYCLE_COUNT, *param_ranges[CHUNK_HEARTCYCLE_COUNT]),
			CHUNK_DURATION: trial.suggest_float(f"cycle_{CHUNK_DURATION}", *param_ranges[CHUNK_DURATION]),
		})
	else:
		params.update({
			AUDIO_LENGTH_NORM: trial.suggest_categorical(AUDIO_LENGTH_NORM, param_ranges[AUDIO_LENGTH_NORM]),
			CHUNK_DURATION: trial.suggest_float(f"fix_{CHUNK_DURATION}", *param_ranges[CHUNK_DURATION]),
		})
	return params

def objective(trial, config, param_ranges):
	update_dict = get_base_config(config)
	update_dict.update(get_beats_knn_params(trial, param_ranges[EMBEDDING_PARAMS]))
	update_dict.update(get_chunk_params(trial, config[CHUNK_METHOD], param_ranges["chunk_params"]))
	update_dict[RUN_NAME_SUFFIX] = f"optuna_{config[TRAIN_DATASET]}_{config[CHUNK_METHOD]}_beats_knn_v2_{trial.number}"

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

def start_optimization(config, param_ranges, n_trials):
	study_name = f"beats_knn_{config[TRAIN_DATASET]}_{config[CHUNK_METHOD]}_v2"
	storage_name = f"sqlite:///{FOLDER_OPTIMIZATION}/optim_survey_3.db"

	try:
		study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True, direction="maximize")
		study.optimize(lambda trial: objective(trial, config, param_ranges), n_trials=n_trials, callbacks=[trial_callback])

		print("Best trial:")
		trial = study.best_trial
		print(f"Value: {trial.value}")
		print("Params:")
		for key, value in trial.params.items():
			print(f"    {key}: {value}")
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
	configs = [
		{
			TRAIN_DATASET: PHYSIONET_2016,
			CHUNK_METHOD: CHUNK_METHOD_FIXED,
		},
		{
			TRAIN_DATASET: PHYSIONET_2022,
			CHUNK_METHOD: CHUNK_METHOD_FIXED,
		},
		{
			TRAIN_DATASET: PHYSIONET_2022,
			CHUNK_METHOD: CHUNK_METHOD_CYCLES,
			AUDIO_LENGTH_NORM: LENGTH_NORM_STRETCH,
		},
	]

	param_ranges = [
		{ # 2016 fix
			EMBEDDING_PARAMS: {
				KNN_N_NEIGHBORS: (1, 13),
				KNN_WEIGHT: [KNN_WEIGHT_UNIFORM],
				KNN_METRIC: [KNN_METRIC_EUCLIDEAN],
				USE_SMOTE: [True],
				USE_UMAP: [False],
				USE_HDBSCAN: [True, False],
				HDBSCAN_PARAM_MIN_CLUSTER_SIZE: (20, 60),
				HDBSCAN_PARAM_MIN_SAMPLES: (3, 10),
				# EMBEDDINGS_REDUCE_UMAP_N_COMPONENTS: (2, 10),
				# EMBEDDINGS_REDUCE_UMAP_N_NEIGHBORS: (5, 30),
				# EMBEDDINGS_REDUCE_UMAP_MIN_DIST: (0.1, 0.5),
			},
			"chunk_params": {
				CHUNK_DURATION: (8.0, 12.0),
				AUDIO_LENGTH_NORM: [LENGTH_NORM_PADDING],
			},
		},
		{	# 22 fix
			EMBEDDING_PARAMS: {
				KNN_N_NEIGHBORS: (19, 35),
				KNN_WEIGHT: [KNN_WEIGHT_UNIFORM, KNN_WEIGHT_DISTANCE],
				KNN_METRIC: [KNN_METRIC_EUCLIDEAN, KNN_METRIC_MANHATTAN],
				USE_SMOTE: [False],
				USE_UMAP: [True, False],
				USE_HDBSCAN: [True],
				EMBEDDINGS_REDUCE_UMAP_N_COMPONENTS: (5, 25),
				EMBEDDINGS_REDUCE_UMAP_N_NEIGHBORS: (10, 60),
				EMBEDDINGS_REDUCE_UMAP_MIN_DIST: (0.05, 0.5),
				HDBSCAN_PARAM_MIN_CLUSTER_SIZE: (15, 50),
				HDBSCAN_PARAM_MIN_SAMPLES: (15, 50),
			},
			"chunk_params": {
				CHUNK_DURATION: (4.0, 7.0),
				AUDIO_LENGTH_NORM: [LENGTH_NORM_PADDING],
			},
		},
		{	 # 22 cycle
			EMBEDDING_PARAMS: {
				KNN_N_NEIGHBORS: (13, 26),
				KNN_WEIGHT: [KNN_WEIGHT_UNIFORM],
				KNN_METRIC: [KNN_METRIC_EUCLIDEAN],
				USE_SMOTE: [False],
				USE_UMAP: [True, False],
				USE_HDBSCAN: [True, False],
				EMBEDDINGS_REDUCE_UMAP_N_COMPONENTS: (7, 25),
				EMBEDDINGS_REDUCE_UMAP_N_NEIGHBORS: (10, 45),
				EMBEDDINGS_REDUCE_UMAP_MIN_DIST: (0.1, 0.4),
				HDBSCAN_PARAM_MIN_CLUSTER_SIZE: (5, 15),
				HDBSCAN_PARAM_MIN_SAMPLES: (15, 25),
			},
			"chunk_params": {
				CHUNK_DURATION: (7.0, 13.0),
				CHUNK_HEARTCYCLE_COUNT: (8, 13),
			},
		},
	]

	# for config, ranges in zip(configs, param_ranges):
	# 	start_optimization(config, ranges, n_trials=10)
	start_optimization(configs[2], param_ranges[2], n_trials=10)
	start_optimization(configs[1], param_ranges[1], n_trials=10)
	start_optimization(configs[0], param_ranges[0], n_trials=7)
