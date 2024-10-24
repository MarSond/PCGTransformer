import os
from pathlib import Path

import yaml

"""
This script is used to extract relevant parameters from the run_config.yaml 
It can be used to generate a summary of all config files in the final_runs folder.
"""

def read_yaml(file_path):
	"""Read YAML file and return contents."""
	with Path(file_path).open() as file:
		return yaml.safe_load(file)

def get_relevant_params(config):
	"""Extract relevant parameters from config."""
	# CNN specific parameters
	if config.get("model_method_type") == "cnn":
		cnn_params = {
			"cnn_activation": config.get("cnn_params", {}).get("activation"),
			"cnn_drop0": config.get("cnn_params", {}).get("drop0"),
			"cnn_drop1": config.get("cnn_params", {}).get("drop1"),
			"cnn_hop_length": config.get("cnn_params", {}).get("hop_length"),
			"cnn_model_sub_type": config.get("cnn_params", {}).get("model_sub_type"),
			"cnn_n_fft": config.get("cnn_params", {}).get("n_fft"),
			"cnn_n_mels": config.get("cnn_params", {}).get("n_mels"),
		}
	else:
		cnn_params = {}
	# Embedding parameters
	if config.get("model_method_type") == "beats":
		emb_params = {
			"knn_n_neighbors": config.get("emb_parameters", {}).get("knn_n_neighbors"),
			"knn_weight": config.get("emb_parameters", {}).get("knn_weight"),
			"emb_use_smote": config.get("emb_parameters", {}).get("emb_use_smote"),
			"emb_use_umap": config.get("emb_parameters", {}).get("emb_use_umap"),
			"use_hdbscan": config.get("emb_parameters", {}).get("use_hdbscan"),
			"hdb_min_cluster_size": config.get("emb_parameters", {}).get("hdb_min_cluster_size"),
			"hdb_min_samples": config.get("emb_parameters", {}).get("hdb_min_samples"),
		}
	else:
		emb_params = {}
	# General training parameters (nicht für KNN)
	if config.get("model_method_type") != "beats":
		training_params = {
			"epochs": config.get("epochs"),
			"l1_regulation_weight": config.get("l1_regulation_weight"),
			"l2_regulation_weight": config.get("l2_regulation_weight"),
			"learning_rate": config.get("learning_rate"),
			"optimizer": config.get("optimizer"),
			"scheduler_factor": config.get("scheduler_factor"),
			"scheduler_patience": config.get("scheduler_patience"),
			"sheduler": config.get("sheduler"),
		}
	else:
		training_params = {}

	# Data processing parameters
	data_params = {
		"training_dataset": config.get("train_dataset"),
		"augmentation_rate": config.get("augmentation_rate"),
		"audio_length_norm": config.get("audio_length_norm"),
		"chunk_duration": config.get("chunk_duration"),
		"chunk_heartcycle_count": config.get("chunk_heartcycle_count"),
		"chunk_method": config.get("chunk_method"),
		"normalization": config.get("normalization"),
	}

	return {
		"data_params": data_params,
		"cnn_params": cnn_params,
		"emb_params": emb_params,
		"training_params": training_params
	}

def create_summary(base_dir):
	"""Create summary of all config files in final_runs directory."""
	summary = {}
	base_path = Path(base_dir)

	# Finde alle YAML Dateien in Unterordnern
	for config_file in base_path.rglob("run_config.yaml"):
		folder_name = config_file.parent.name
		config = read_yaml(config_file)
		summary[folder_name] = get_relevant_params(config)

	return summary

def write_summary(summary, output_file):
	"""Write summary to file in a machine-readable format."""
	with Path(output_file).open("w") as f:
		for folder, params in summary.items():
			f.write(f"[{folder}]\n")

			for section, section_params in params.items():
				if len(section_params) > 0:
					f.write(f"{section}:\n")
					for param, value in section_params.items():
						try:
							f.write(f"{param}={value}\n")
						except UnicodeEncodeError:
							f.write(f"unicode error\n")
					f.write(f"\n")

			f.write(f"{'-'*80}\n") # Leerzeile zwischen Ordnern

def main():
	final_runs_dir = "final_runs"
	output_file = Path(final_runs_dir) / "config_summary.txt"

	summary = create_summary(final_runs_dir)
	write_summary(summary, output_file)

if __name__ == "__main__":
	main()
