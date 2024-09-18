from MLHelper.constants import *
from run import Run


def do_run(config: dict):
	run = Run(config_update_dict=config)
	run.setup_task()
	result = run.start_task()
	print(result)


if __name__ == "__main__":

	train_update_dict = {
		TASK_TYPE: TRAINING, METADATA_FRAC: 0.05,
		CNN_PARAMS: {},
		EPOCHS: 50, BATCH_SIZE: 80,
		SINGLE_BATCH_MODE: False, TRAIN_FRAC: 0.8, KFOLD_SPLITS: 1,
		# TRAINING_CHECKPOINT: {EPOCH: 70, RUN_NAME: "run1", FOLD: 6},
		CHUNK_DURATION: 7.0, CHUNK_METHOD: CHUNK_METHOD_FIXED,
		DO_FAKE_UPDATES: 0, RUN_NAME_SUFFIX: "beats-test",
		EARLY_STOPPING_ENABLED: False,
		MODEL_METHOD_TYPE: CNN,
		SAVE_MODEL: False,
	}

	knn_base_dict = train_update_dict.copy()
	knn_base_dict.update({
		MODEL_METHOD_TYPE: BEATS,
		EPOCHS: 1, OPTIMIZER: None, SCHEDULER: None,
		TRANSFORMER_PARAMS: {MODEL_SUB_TYPE: MODEL_TYPE_KNN},
	})


	# General test for KNN embeddings mode
	knn_test_dict = knn_base_dict.copy()
	knn_test_dict.update({
		TRAIN_DATASET: PHYSIONET_2022, KFOLD_SPLITS: 1,
		METADATA_FRAC: 1.0,
		EMBEDDING_PARAMS: {
			KNN_N_NEIGHBORS: 1,
			KNN_WEIGHT: KNN_WEIGHT_UNIFORM,
			KNN_METRIC: KNN_METRIC_EUCLIDEAN,
			KNN_ALGORITHM: KNN_ALGORITHM_AUTO,
			USE_SMOTE: True,
			USE_HDBSCAN: True,
			REDUCE_DIM_UMAP: True,
			EMBEDDINGS_REDUCE_UMAP_MIN_DIST: 0.1,
			EMBEDDINGS_REDUCE_UMAP_N_NEIGHBORS: 15,
			EMBEDDINGS_REDUCE_UMAP_N_COMPONENTS: 10,
		},
	})
	do_run(knn_test_dict)
	####

	########## To extract all embeddings for further training ##########
	knn_extract_2016_fix = knn_base_dict.copy()
	knn_extract_2016_fix.update({
		TRAIN_DATASET: PHYSIONET_2016,
		KFOLD_SPLITS: 1,
		METADATA_FRAC: 1.0,
		SAVE_MODEL: True,
		CHUNK_METHOD: CHUNK_METHOD_FIXED,
		AUDIO_LENGTH_NORM: LENGTH_NORM_PADDING,
		CHUNK_DURATION: 9.0,
		RUN_NAME_SUFFIX: "2016_fix_optim_audio",
	})
	#do_run(knn_extract_2016_fix)

	knn_extract_2022_fix = knn_base_dict.copy()
	knn_extract_2022_fix.update({
		TRAIN_DATASET: PHYSIONET_2022,
		KFOLD_SPLITS: 1,
		METADATA_FRAC: 1.0,
		SAVE_MODEL: True,
		AUDIO_LENGTH_NORM: LENGTH_NORM_PADDING,
		CHUNK_METHOD: CHUNK_METHOD_FIXED,
		CHUNK_DURATION: 5.0,
		RUN_NAME_SUFFIX: "2022_fix_optim_audio",
	})
	#do_run(knn_extract_2022_fix)

	knn_extract_2022_cycle = knn_base_dict.copy()
	knn_extract_2022_cycle.update({
		TRAIN_DATASET: PHYSIONET_2022,
		KFOLD_SPLITS: 1,
		METADATA_FRAC: 1.0,
		SAVE_MODEL: True,
		AUDIO_LENGTH_NORM: LENGTH_NORM_STRETCH,
		CHUNK_METHOD: CHUNK_METHOD_CYCLES,
		CHUNK_HEARTCYCLE_COUNT: 12,
		CHUNK_DURATION: 9.0,
		RUN_NAME_SUFFIX: "2022_cycle_optim_audio",
	})
	#do_run(knn_extract_2022_cycle)

	##### To test loading from ebeddings
	knn_continue_emb_dict = knn_base_dict.copy()
	knn_continue_emb_dict.update({
		TRAIN_DATASET: PHYSIONET_2016,
		KFOLD_SPLITS: 1,
		METADATA_FRAC: 1.0,
		CHUNK_DURATION: 9.0,
		CHUNK_METHOD: CHUNK_METHOD_FIXED,
		AUDIO_LENGTH_NORM: LENGTH_NORM_PADDING,
		LOAD_EMBEDDINGS_FROM_RUN_NAME: "2024-09-17_15-50-38_2016_fix_optim_audio",
	})
	#do_run(knn_continue_emb_dict)


######

	##### To test fold and epoch logic
	fold_test_dict = train_update_dict.copy()
	fold_test_dict.update({
		MODEL_METHOD_TYPE: CNN,
		TRAIN_DATASET: PHYSIONET_2016, KFOLD_SPLITS: 3, EPOCHS: 2,
	})
	#do_run(fold_test_dict)


	##### To test continue from a saved model checkpoint
	continue_test = {
		METADATA_FRAC: 0.05, RUN_NAME_SUFFIX: "continue-test", EPOCHS: 82,
		TRAINING_CHECKPOINT: {EPOCH: 80, RUN_NAME: "2024-07-12_21-52-51_combined-optimized-2", FOLD: 1},
		LOAD_PREVIOUS_RUN_NAME: "2024-07-12_21-52-51_combined-optimized-2",
	}

	# do_run(continue_test)

# Run names to use:
# 2024-09-17_12-59-02_2016_fix_optim_audio
# 2024-09-17_13-02-44_2022_fix_optim_audio
# 2024-09-17_13-05-43_2022_cycle_optim_audio