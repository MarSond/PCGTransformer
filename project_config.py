from MLHelper.constants import *

project_config = {
	DATA_BASE_PATH: "data/",
	LABEL_NAME: "label_1",
	RUN_FOLDER: "runs/",
	FILENAME_RUN_CONFIG: FILENAME_RUN_CONFIG_VALUE,
	FILENAME_LOG_OUTPUT: FILENAME_LOG_OUTPUT_VALUE,
	RUN_RESULTS_PATH: "",		# Path to save run results
	LOAD_PREVIOUS_RUN_NAME: "",	# if not empty, load config from named run inside runs folder
	CHUNK_DURATION: 7.0,		# Duration in seconds of each chunked audio split
	TASK_TYPE: TASK_TYPE_TRAINING,
	RUN_NAME: "blank_name",
	EPOCHS: 30,					# Number of epochs to train

	BATCH_SIZE: 32,				# Batch size
	METADATA_FRAC: 1.0,			# Percentage of data to use in the run
	TRAIN_FRAC: 0.8,			# Fraction of data to use for training, when using kfold=1
	SINGLE_BATCH_MODE: False,	# End of training after one batch, for testing
	MODEL_METHOD_TYPE: CNN,		# method to use for training: CNN, BEATs
	TRAIN_DATASET: PHYSIONET_2016,
	INFERENCE_DATASET: PHYSIONET_2016,
	INFERENCE_MODEL: None,
	TRAINING_CHECKPOINT: None,		# load checkpoint for training: {EPOCH: 80, RUN_NAME: "run1", FOLD: 1}
	AUGMENTATION_RATE: 0.0,			# Percent of data passed to data augmentation pipeline
	DATALOADER_NUM_WORKERS: 0,		# workers for pytorch dataloader
	KFOLD_SPLITS: 1,				# number of kfold splits. 1 -> use normal splitting
	SAVE_MODEL: True,				# Save pytorch model
	SAVE_ONLY_LAST_MODEL: True,		# Save only last model (deletes old models when saving new)
	PLOT_METRICS: True,				# Plot metrics during training
	DELETE_OWN_RUN_FOLDER: False,	# Delete own run folder when finished

	CHUNK_METHOD: CHUNK_METHOD_FIXED,		# Chunking method to split longer files into smaller chunks
	CHUNK_HEARTCYCLE_COUNT: 5,				# Number of heart cycles per chunk when using CHUNK_METHOD_CYCLES
	CHUNK_PADDING_THRESHOLD: 0.60,			# minimum duaration of a full chunk required to be considered
	NORMALIZATION: NORMALIZATION_MAX_ABS,	# audio normalisation method
	AUDIO_LENGTH_NORM: LENGTH_NORM_PADDING,	# Audio length normalisation method

	GRAD_ACCUMULATE_STEPS: 1,				# Number of gradient accumulation steps
	DO_FAKE_UPDATES: 0,						# Enable fake updates values in metric manager (for testing purposes)

	EARLY_STOPPING_ENABLED: True,			# Enable early stopping after validation metrics
	EARLY_STOPPING_PATIENCE: 8,				# Count of epochs to wait before checking early stopping
	EARLY_STOPPING_THRESHOLD: 0.60,			# Threshold for early stopping (minimum required)
	EARLY_STOPPING_METRIC: METRICS_NMCC,	# Metric to use for early stopping

	USE_AMP : True,					# Enable Automatic Mixed Precision
	OPTIMIZER: OPTIMIZER_ADAM,		# OPTIMIZER_ADAM, OPTIMIZER_SGD
	SCHEDULER: SCHEDULER_PLATEAU,	# SCHEDULER_COSINE, SCHEDULER_STEP, SCHEDULER_PLATEAU, None
	SCHEDULER_PATIENCE: 10,
	LEARNING_RATE: 0.001,			# initial learning rate
	SCHEDULER_FACTOR: 0.1,

	L1_REGULATION_WEIGHT: 0.0,		# L1 weight decay, used with raw loss in step function # TODO test 0 or None as default
	L2_REGULATION_WEIGHT: 0.000001,	# L2 weight decay, passed to optimizer
	LOSS_FUNCTION: LOSS_FOCAL_LOSS,	# LOSS_CROSS_ENTROPY, LOSS_FOCAL_LOSS
	LOSS_FUNCTION_PARAMETER_1: 2.0,	# Focal Loss: gamma,
	LOSS_FUNCTION_PARAMETER_2: None,

	SIGNAL_FILTER: BUTTERPASS,		# BUTTERPASS, NONE
	BUTTERPASS_LOW: 25,				# Low pass filter for audio signal
	BUTTERPASS_HIGH: 500,			# High pass filter for audio signal
	BUTTERPASS_ORDER: 5,			# Order of the butterworth filter

	USE_PROFILER: False,	# Schalter für das Profiling
	PROFILER_EPOCHS: 1,		# Anzahl der Epochen, die profiliert werden sollen
	PROFILER_STEPS: 100,	# Anzahl der Schritte pro Epoche, die profiliert werden sollen

	CNN_PARAMS: {			# Values individual for CNN methods
		DROP0: 0.5,			# Dropout rate on position 0
		DROP1: 0.2,			# Dropout rate on position 1
		ACTIVATION: ACTIVATION_SILU, # Activation function
		N_MELS: 512,
		HOP_LENGTH: 128,
		N_FFT: 1024,
		TOP_DB: 80.0,
		MODEL_SUB_TYPE: 1,
	},
	TRANSFORMER_PARAMS: {
		ACTIVATION: ACTIVATION_SILU,
		FREEZE_EXTRACTOR: True,	# Freeze the beats model
		DROP0: 0.5,				# Dropout rate on position 0
		DROP1: 0.3,				# Dropout rate on position 1
		MODEL_SUB_TYPE: 1, 		# Specific model impementation for BEATs type
		EXTRACTOR_FOLDER: f"beats_classifier/{EXTRACTOR_FOLDER_NAME}/",
		EXTRACTOR_NAME: "BEATs_iter3_plus_AS2M.pt",	# Name of the model for extracting
	},
	EMBEDDING_PARAMS: {
		EMBEDDING_CLASSIFIER: CLASSIFIER_KNN,	# CLASSIFIER_KNN, CLASSIFIER_SVM (not implemented yet), ...
		USE_SMOTE: False,						# Enable SMOTE on the fly
		EMBEDDINGS_REDUCE_UMAP_N_COMPONENTS: 2,	# Number of components for UMAP, if enabled
		EMBEDDINGS_REDUCE_UMAP_N_NEIGHBORS: 5,	# Number of neighbors for UMAP, if enabled
		KNN_N_NEIGHBORS: 5,						# Number of neighbors for KNN
		KNN_METRIC: KNN_METRIC_EUCLIDEAN,		# KNN_METRIC_EUCLIDEAN, KNN_METRIC_MANHATTAN
		KNN_WEIGHT: KNN_WEIGHT_UNIFORM,			# KNN_WEIGHT_UNIFORM, KNN_WEIGHT_DISTANCE
		EMBEDDING_COMBINE_METHOD: EMBEDDING_COMBINE_METHOD_MEAN, # method to combine, if more than 1 Set per input
		KNN_ALGORITHM: KNN_ALGORITHM_AUTO,		# KNN_ALGORITHM_AUTO, KNN_ALGORITHM_KD_TREE
		USE_UMAP: False,					# Reduce dimensionality with UMAP
		USE_HDBSCAN: False,						# Enable HDBSCAN
		HDBSCAN_PARAM_MIN_CLUSTER_SIZE: 5,		# Minimum cluster size for HDBSCAN, if enabled
		HDBSCAN_PARAM_MIN_SAMPLES: 5,			# Minimum samples for HDBSCAN, if enabled
		EMBEDDING_SAVE_TO_FILE: True,			# Save embeddings to file
		EMBEDDING_PLOT_UMAP: True,				# Plot UMAP before and after fitting
	},
	LOAD_EMBEDDINGS_FROM_RUN_NAME: None,		# Run name to load embeddings from, skips extraction
}
