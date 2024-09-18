from pathlib import Path

import matplotlib.pyplot as plt

from MLHelper.constants import *
from MLHelper.tools.utils import MLUtil
from run import Run

# ruff: noqa: T201

def send_result_mail(name: str, results: dict):
	try:
		subject = f"Study Complete: {name}"
		body = f"Study completed successfully.\n\nStudy name: {name}\nresults:\n{results}"
		own_adress = "martinsondermann10@gmail.com"

		if MLUtil.send_self_mail_gmail(subject, body, own_adress):
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
		send_result_mail(config[RUN_NAME_SUFFIX], result)
	except Exception as e:
		print(e)
		print("Failed to start training.")


if __name__ == "__main__":

	base_config = {
		TASK_TYPE: TRAINING, METADATA_FRAC: 1.0,
		SINGLE_BATCH_MODE: False, KFOLD_SPLITS: 10,
		SAVE_ONLY_LAST_MODEL: False, SAVE_MODEL: True,
		EARLY_STOPPING_ENABLED: False,
		EPOCHS: 60,
	}

	knn_config = base_config.copy()
	knn_config.update({
		EPOCHS: 1,
		BATCH_SIZE: 16,
		MODEL_METHOD_TYPE: BEATS,
		TRANSFORMER_PARAMS: {
			MODEL_SUB_TYPE: MODEL_TYPE_KNN,
		},

	})

	# 2016 Fix CNN
	physionet_2016_fixed_cnn = base_config.copy()
	physionet_2016_fixed_cnn.update({
		TRAIN_DATASET: PHYSIONET_2016,
		RUN_NAME_SUFFIX: "2016_fixed_cnn_finalrun",
		MODEL_METHOD_TYPE: CNN,
		CHUNK_METHOD: CHUNK_METHOD_FIXED,
		# 2024-08-31_08-38-31_2016_fixed_cnn_fullrun-final3
		CHUNK_DURATION: 10.0,
		OPTIMIZER: OPTIMIZER_ADAMW,
		LEARNING_RATE: 0.0001,
		SCHEDULER: SCHEDULER_PLATEAU,
		SCHEDULER_PATIENCE: 10,
		SCHEDULER_FACTOR: 0.5,
		AUGMENTATION_RATE: 0.6,
		BATCH_SIZE: 72,
		L1_REGULATION_WEIGHT: 1.0e-05,
		L2_REGULATION_WEIGHT: 0.0005,
		CNN_PARAMS: {
			ACTIVATION: ACTIVATION_SILU,
			DROP0: 0.3,
			DROP1: 0.6,
			N_MELS: 512,
			HOP_LENGTH: 128,
			N_FFT: 512,
			MODEL_SUB_TYPE: 2,
		}
	})

	# 2016 Fix BEATS kNN
	physionet_2016_fixed_beats_knn = knn_config.copy()
	physionet_2016_fixed_beats_knn.update({
		TRAIN_DATASET: PHYSIONET_2016,
		RUN_NAME_SUFFIX: "2016_fixed_beats_knn_finalrun",
		CHUNK_METHOD: CHUNK_METHOD_FIXED,
		CHUNK_DURATION: 10.0,
		EMBEDDING_PARAMS: {
			KNN_N_NEIGHBORS: 5,
			KNN_WEIGHT: KNN_WEIGHT_UNIFORM,
			KNN_METRIC: KNN_METRIC_EUCLIDEAN,
			USE_SMOTE: True,
			USE_HDBSCAN: False,
			USE_UMAP: False,
		}
	})

	# 2022 Fix CNN
	physionet_2022_fixed_cnn = base_config.copy()
	physionet_2022_fixed_cnn.update({
		TRAIN_DATASET: PHYSIONET_2022,
		RUN_NAME_SUFFIX: "2022_fixed_cnn_finalrun",
		MODEL_METHOD_TYPE: CNN,
		CHUNK_METHOD: CHUNK_METHOD_FIXED,
		CHUNK_DURATION: 10.0,
		OPTIMIZER: OPTIMIZER_ADAM,
		LEARNING_RATE: 1e-4,
		SCHEDULER: SCHEDULER_PLATEAU,
		SCHEDULER_PATIENCE: 10,
		SCHEDULER_FACTOR: 0.5,
		AUGMENTATION_RATE: 0.6,
		BATCH_SIZE: 72,
		L1_REGULATION_WEIGHT: 0.005,
		L2_REGULATION_WEIGHT: 2.8e-07,
		CNN_PARAMS: {
			ACTIVATION: ACTIVATION_SILU,
			DROP0: 0.4,
			DROP1: 0.6,
			N_MELS: 384,
			HOP_LENGTH: 352,
			N_FFT: 384,
			MODEL_SUB_TYPE: 2,
		}
	})

	# 2022 Fix BEATS Linear
	physionet_2022_fixed_beats_linear = base_config.copy()
	physionet_2022_fixed_beats_linear.update({
		TRAIN_DATASET: PHYSIONET_2022,
		MODEL_METHOD_TYPE: BEATS,
		CHUNK_METHOD: CHUNK_METHOD_FIXED,
		RUN_NAME_SUFFIX: "2022_fixed_beats_linear_finalrun",
		CHUNK_DURATION: 12.0,
		OPTIMIZER: OPTIMIZER_ADAMW,
		LEARNING_RATE: 0.0001,
		SCHEDULER: SCHEDULER_PLATEAU,
		SCHEDULER_PATIENCE: 10,
		SCHEDULER_FACTOR: 0.5,
		AUGMENTATION_RATE: 0.6,
		BATCH_SIZE: 8,
		GRAD_ACCUMULATE_STEPS: 7,
		L1_REGULATION_WEIGHT:  0.0001,
		L2_REGULATION_WEIGHT: 0.005,
		TRANSFORMER_PARAMS: {
			ACTIVATION: ACTIVATION_RELU,
			DROP0: 0.4,
			DROP1: 0.6,
			MODEL_SUB_TYPE: 2,
		}
	})

	# 2022 Fix BEATS kNN
	physionet_2022_fixed_beats_knn = knn_config.copy()
	physionet_2022_fixed_beats_knn.update({
		TRAIN_DATASET: PHYSIONET_2022,
		RUN_NAME_SUFFIX: "2022_fixed_beats_knn_finalrun",
		CHUNK_METHOD: CHUNK_METHOD_FIXED,
		CHUNK_DURATION: 5.0,
		EMBEDDING_PARAMS: {
			KNN_N_NEIGHBORS: 23,
			KNN_WEIGHT: KNN_WEIGHT_DISTANCE,
			KNN_METRIC: KNN_METRIC_EUCLIDEAN,
			USE_SMOTE: False,
			USE_HDBSCAN: False,
			USE_UMAP: False,
		}
	})

	# 2022 Cycles CNN
	physionet_2022_cycles_cnn = base_config.copy()
	physionet_2022_cycles_cnn.update({
		TRAIN_DATASET: PHYSIONET_2022,
		RUN_NAME_SUFFIX: "2022_cycles_cnn_finalrun",
		MODEL_METHOD_TYPE: CNN,
		CHUNK_METHOD: CHUNK_METHOD_CYCLES,
		CHUNK_HEARTCYCLE_COUNT: 7,
		AUGMENTATION_RATE: 0.6,
		CHUNK_DURATION: 10.0,
		OPTIMIZER: OPTIMIZER_ADAMW,
		LEARNING_RATE: 5.0e-5,
		L1_REGULATION_WEIGHT: 0.0,
		L2_REGULATION_WEIGHT: 0.001,
		SCHEDULER: SCHEDULER_PLATEAU,
		CNN_PARAMS: {
			ACTIVATION: ACTIVATION_RELU,
			DROP0: 0.4,
			DROP1: 0.6,
			N_MELS: 128,
			HOP_LENGTH: 288,
			N_FFT: 1152,
			MODEL_SUB_TYPE: 4,
		}
	})

	# 2022 Cycles BEATS kNN
	physionet_2022_cycles_beats_knn = knn_config.copy()
	physionet_2022_cycles_beats_knn.update({
		TRAIN_DATASET: PHYSIONET_2022,
		RUN_NAME_SUFFIX: "2022_cycles_beats_knn_finalrun",
		CHUNK_METHOD: CHUNK_METHOD_CYCLES,
		CHUNK_HEARTCYCLE_COUNT: 12,
		CHUNK_DURATION: 8.0,
		EMBEDDING_PARAMS: {
			KNN_N_NEIGHBORS: 21,
			KNN_WEIGHT: KNN_WEIGHT_UNIFORM,
			KNN_METRIC: KNN_METRIC_EUCLIDEAN,
			USE_SMOTE: True,
			USE_HDBSCAN: False,
			USE_UMAP: True,
			EMBEDDINGS_REDUCE_UMAP_MIN_DIST: 0.1,
			EMBEDDINGS_REDUCE_UMAP_N_NEIGHBORS: 10,
			EMBEDDINGS_REDUCE_UMAP_N_COMPONENTS: 7,
		}
	})

	# Execute runs
	models_to_run = [
		physionet_2016_fixed_beats_knn,		# started
		physionet_2022_fixed_beats_knn,
		physionet_2022_cycles_beats_knn
		physionet_2016_fixed_cnn,			# started
		physionet_2022_fixed_cnn,			# started
		physionet_2022_cycles_cnn,			# started
		physionet_2022_fixed_beats_linear,	# started
	]

	# TODO wenn ergebnisse nicht zufriedenstellend: Stumpf bestes ergebnis nachtrainieren

	# TODO test mit force HDB scan
	# HDB scan ganz weglassen
	# Wenn doch: outlier analyse manuell später durchführen
	for model_config in models_to_run:
		do_run(model_config)

	plt.show()
