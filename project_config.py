from MLHelper.constants import *

project_config = {
	DATA_BASE_PATH: "data/",
	LABEL_NAME: "label_1",
	RUN_FOLDER: "runs/",
	FILENAME_RUN_CONFIG: FILENAME_RUN_CONFIG_VALUE,
	FILENAME_LOG_OUTPUT: FILENAME_LOG_OUTPUT_VALUE,
	LOAD_PREVIOUS_RUN_NAME: "", # if not empty, load config from named run inside runs folder
	CHUNK_DURATION: 7.0,	# Duration in seconds of each chunked audio split
	TASK_TYPE: TASK_TYPE_TRAINING,
	RUN_NAME: "blank_name",
	METADATA_FRAC: 1.0, # Percentage of data to use in the run
	TRAIN_FRAC: 0.8, # Fraction of data to use for training, when using kfold=1
	SINGLE_BATCH_MODE: False,	# End of training after one batch, for testing
	MODEL_METHOD_TYPE: CNN,	# method to use for training: CNN, BEATs
	TRAIN_DATASET: PHYSIONET_2016,
	INFERENCE_DATASET: PHYSIONET_2016,
	INFERENCE_MODEL: None,
	TRAINING_CHECKPOINT: None,	# load checkpoint for training: {EPOCH: 80, RUN_NAME: "run1", FOLD: 1}
	AUGMENTATION_RATE: 0.0,
	NUM_WORKERS: 0,			# workers for dataloader
	KFOLD_SPLITS: 5,		# number of kfold splits. 1 -> use normal splitting
	SAVE_MODEL: True,		# Save pytorch model
	SAVE_ONLY_LAST_MODEL: True, # Save only last model (deletes old models when saving new)
	CHUNK_METHOD: CHUNK_METHOD_FIXED, # Chunking method to split longer files into smaller chunks
	CHUNK_HEARTCYCLE_COUNT: 5,	# Number of heart cycles per chunk when using CHUNK_METHOD_CYCLES
	CHUNK_PADDING_THRESHOLD: 0.65, # minimum duaration of a full chunk required to be considered
	NORMALIZATION: NORMALIZATION_MAX_ABS, # audio normalisation method
	EPOCHS: 30,	# Count of epochs to train
	BATCH_SIZE: 72, # Batch size
	GRAD_ACCUMULATE_STEPS: 1,
	EARLY_STOPPING_ENABLED: True, # Enable early stopping after validation metrics
	EARLY_STOPPING_PATIENCE: 8, # Count of epochs to wait before checking early stopping
	EARLY_STOPPING_THRESHOLD: 0.60, # Threshold for early stopping (minimum required)
	EARLY_STOPPING_METRIC: METRICS_NMCC, # Metric to use for early stopping
	AUDIO_LENGTH_NORM: LENGTH_NORM_REPEAT, # Audio length normalisation method
	DO_FAKE_UPDATES: 0, # Enable fake updates values in metric manager (for testing purposes)
	RUN_RESULTS_PATH: "", # Path to save run results
	USE_AMP : True, # Enable Automatic Mixed Precision
	OPTIMIZER: OPTIMIZER_ADAM, # OPTIMIZER_ADAM, OPTIMIZER_SGD
	SCHEDULER: SCHEDULER_PLATEAU, # SCHEDULER_COSINE, SCHEDULER_STEP, SCHEDULER_PLATEAU, None
	LEARNING_RATE: 0.001, # initial learning rate
	L1_REGULATION_WEIGHT: 0.000001,
	L2_REGULATION_WEIGHT: 0.000001,
	LOSS_FUNCTION: LOSS_FOCAL_LOSS, # LOSS_CROSS_ENTROPY, LOSS_FOCAL_LOSS
	LOSS_FUNCTION_PARAMETER_1: 2.0, # Focal Loss: gamma,
	LOSS_FUNCTION_PARAMETER_2: None,
	SCHEDULER_PATIENCE: 10,
	SCHEDULER_FACTOR: 0.1,
	SIGNAL_FILTER: BUTTERPASS,
	BUTTERPASS_LOW: 25,
	BUTTERPASS_HIGH: 500,
	BUTTERPASS_ORDER: 5,
	USE_PROFILER: False,  # Schalter für das Profiling
	PROFILER_EPOCHS: 1,   # Anzahl der Epochen, die profiliert werden sollen
	PROFILER_STEPS: 100,  # Anzahl der Schritte pro Epoche, die profiliert werden sollen
	CNN_PARAMS: {	# Values individual for CNN methods
		DROP0: 0.5, # Dropout rate on position 0
		DROP1: 0.2,	# Dropout rate on position 1
		ACTIVATION: ACTIVATION_SILU, # Activation function
		N_MELS: 512,
		HOP_LENGTH: 128,
		N_FFT: 1024,
		TOP_DB: 80.0,
		BATCHNORM: True, # Use batchnorm (# TODO not implemented checking for)
		MODEL_SUB_TYPE: 1,
	},
	TRANSFORMER_PARAMS: {
		FREEZE_EXTRACTOR: True, # Freeze the beats model
		ACTIVATION: ACTIVATION_SILU,
		DROP0: 0.5, # Dropout rate on position 0
		DROP1: 0.3,	# Dropout rate on position 1
		EXTRACTOR_FOLDER: f"beats_classifier/{EXTRACTOR_FOLDER_NAME}/",
		EXTRACTOR_NAME: "BEATs_iter3_plus_AS2M.pt",	# Name of the model for extracting
		MODEL_SUB_TYPE: 1, # Specific model impementation for BEATs type
	}
}
