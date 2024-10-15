import json
import time
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch

from beats_classifier.beats_dataset import BEATsDataset
from cnn_classifier.cnn_dataset import CNN_Dataset
from MLHelper import embedding_model
from MLHelper.audio.audioutils import AudioUtil
from MLHelper.constants import *
from MLHelper.tools.utils import FileUtils, MLModelInfo, MLPbar, MLUtil
from run import Run, TaskBase

# ruff: noqa: T201

def get_latest_model_path(run_path: Path) -> Path:
	model_folder = run_path / MODEL_FOLDER
	model_files = list(model_folder.glob(f"*.{MODEL_FILE_EXTENSION}"))
	return max(model_files, key=lambda x: x.stat().st_mtime)

def create_and_load_model(run: Run, model_path: str):
	config = run.config
	if config[MODEL_METHOD_TYPE] == BEATS and config[TRANSFORMER_PARAMS].get(MODEL_SUB_TYPE) == MODEL_TYPE_EMBEDDING:
		extractor = TaskBase.create_new_model(run)
		knn_params = config[EMBEDDING_PARAMS]
		model = embedding_model.EmbeddingClassifier(extractor, knn_params, run.logger_dict[LOGGER_TENSOR], run.device)
		state_dict = torch.load(model_path, map_location=run.device)
		model.load_state_dict(state_dict)
	else:
		model = TaskBase.create_new_model(run)
		model, _, _, _ = MLUtil.load_model(model=model, device=run.device, path=model_path)
	return model.to(run.device)

def run_benchmark(run_path: Path, num_runs: int = 11, warmup: int = 1):
	results = {"training": [], "inference": []}

	config = {
		TASK_TYPE: TRAINING,
		LOAD_PREVIOUS_RUN_NAME: run_path.name,
		SINGLE_BATCH_MODE: False,
		KFOLD_SPLITS: 1,
		BATCH_SIZE: 2,
		SAVE_MODEL: False,
		EPOCHS: 1,
		METADATA_FRAC: 1.0,
		TRAIN_FRAC: 1.0,
		PLOT_METRICS: False,
		DELETE_OWN_RUN_FOLDER: True,
		EMBEDDING_PARAMS: {
			EMBEDDING_PLOT_UMAP: False,
			EMBEDDING_SAVE_TO_FILE: False
		}
	}

	run = Run(config_update_dict=config)
	run.setup_task()

	# Load the model for inference
	model_path = get_latest_model_path(run_path)
	model = create_and_load_model(run, str(model_path))

	def perform_round() -> int:
		run.task.trainer_class.set_training_utilities(start_model=model, \
			optimizer=run.task.optimizer, scheduler=run.task.scheduler, scaler=run.task.scaler)
		start_time = time.time()
		run.task.trainer_class.kfold_loop()
		end_time = time.time()
		total_time = (end_time - start_time) * 1000  # in ms
		time_per_example = total_time / (len(run.task.trainer_class.train_loader) * config[BATCH_SIZE])
		return time_per_example

	# Warmup
	for _ in range(warmup):
		perform_round()

	# Benchmark runs
	for _ in range(num_runs - warmup):
		# Training benchmark
		training_time = perform_round()
		results["training"].append(training_time)

	if run.config[MODEL_METHOD_TYPE] == CNN:
		dataset_class = CNN_Dataset
	elif run.config[MODEL_METHOD_TYPE] == BEATS:
		dataset_class = BEATsDataset
	dataset = dataset_class(datalist=run.task.dataset.chunk_list, run=run)
	dataset.set_mode(VALIDATION)
	model.eval()
	for round in range(num_runs - warmup):
		print(f"start round {round}")
		start_time = time.time()
		for idx in range(len(dataset)):
			instance = dataset[idx]  # Dies ruft intern handle_instance auf
			inputs = instance[0]  # Nehme an, dass das erste Element die Eingabedaten sind
			if isinstance(inputs, torch.Tensor):
				inputs = inputs.unsqueeze(0).to(run.device)
			elif isinstance(inputs, np.ndarray):
				inputs = torch.from_numpy(inputs).unsqueeze(0).to(run.device)
			# Überprüfen und anpassen der Eingabeform

			#inputs = inputs.unsqueeze(0)  # Füge Batch-Dimension hinzu

			with torch.no_grad():
				_ = model(inputs)
		end_time = time.time()
		total_time = (end_time - start_time) * 1000  # in ms
		time_per_example = total_time / len(dataset)
		results["inference"].append(time_per_example)
		print(f"end round {round}")

	results["model_size"] = MLModelInfo.get_model_size(model)
	results["file_size_mb"] = FileUtils.get_file_size(model_path)

	return results, run.config

def format_results(results: dict) -> str:
	output = []
	for phase in ["training", "inference"]:
		times = results[phase]
		avg_time = mean(times)
		std_time = stdev(times) if len(times) > 1 else 0
		output.append(f"{phase.capitalize()}:")
		output.append(f"  Average time: {avg_time:.2f} ms/example")
		output.append(f"  Standard deviation: {std_time:.2f} ms")
	return "\n".join(output)

def benchmark_model(run_path: Path):
	print(f"\nBenchmarking Model: {run_path.name}")
	results, config = run_benchmark(run_path)
	try:
		print(format_results(results))
	except Exception as e:
		pass

	torch.cuda.empty_cache()
	return {
		"run_name": run_path.name,
		"training": {
			"mean": np.mean(results["training"]),
			"std": np.std(results["training"])
		},
		"inference": {
			"mean": np.mean(results["inference"]),
			"std": np.std(results["inference"])
		},
		"model_size": results["model_size"],
		"file_size_mb": results["file_size_mb"],
	}


if __name__ == "__main__":
	final_runs_path = Path("final_runs")
	all_results = []
	for run_path in final_runs_path.iterdir():
		if "2024-09-19_01-01-49_2016_fixed_cnn_finalrun" not in run_path.name:
			pass
			#continue
		if run_path.is_dir():
			result = benchmark_model(run_path)
			all_results.append(result)
	print(f"\nResults for {len(all_results)} runs:")
	# Save all results in a single file
	results_path = Path("documents") / "benchmark_results.json"
	results_path.parent.mkdir(parents=True, exist_ok=True)
	with open(results_path, "w") as f:
		json.dump(all_results, f, indent=2)
	print(f"All results saved to {results_path}")
