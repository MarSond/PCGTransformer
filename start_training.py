import matplotlib.pyplot as plt

from MLHelper.constants import *
from run import Run


def do_run(config: dict):
	try:
		run = Run(config_update_dict=config)
		run.setup_task()
		run.start_task()
	except Exception as e:
		print(e)


if __name__ == "__main__":

	train_update_dict = {	TASK_TYPE: TRAINING, METADATA_FRAC: 1.0, \
							CNN_PARAMS: {}, EPOCHS: 30, \
							SINGLE_BATCH_MODE: False, TRAIN_FRAC: 0.8, KFOLD_SPLITS: 1, \
							# TRAINING_CHECKPOINT: {EPOCH: 70, RUN_NAME: "run1", FOLD: 6}, \
							CHUNK_DURATION: 7.0, CHUNK_METHOD: CHUNK_METHOD_FIXED, \
                      }

	cycle_base = train_update_dict.copy()
	cycle_base.update({TRAIN_DATASET: PHYSIONET_2022, CHUNK_METHOD: CHUNK_METHOD_CYCLES, \
		AUDIO_LENGTH_NORM: LENGTH_NORM_STRETCH})

	run_2022 = train_update_dict.copy()
	run_2022.update({TRAIN_DATASET: PHYSIONET_2022, RUN_NAME_SUFFIX: "run2022"})
	do_run(run_2022)

	run_2016 = train_update_dict.copy()
	run_2016.update({TRAIN_DATASET: PHYSIONET_2016, RUN_NAME_SUFFIX: "run2016"})
	do_run(run_2016)

	run2016_2022 = train_update_dict.copy()
	run2016_2022.update({TRAIN_DATASET: PHYSIONET_2016_2022, RUN_NAME_SUFFIX: "run2022-2016"})
	do_run(run2016_2022)
	###


	run_10c_7s = cycle_base.copy()
	run_10c_7s.update({CHUNK_DURATION: 7.0, CHUNK_HEARTCYCLE_COUNT: 10, RUN_NAME_SUFFIX: "cycles-10"})
	#do_run(run_10c_7s)

	run5_5c_5s = cycle_base.copy()
	run5_5c_5s.update({CHUNK_DURATION: 5.0, CHUNK_HEARTCYCLE_COUNT: 5, RUN_NAME_SUFFIX: "cycles-5"})
	#do_run(run5_5c_5s)


	run_5c_3s = cycle_base.copy()
	run_5c_3s.update({
		RUN_NAME_SUFFIX: "5c-3s-m2",
		CHUNK_DURATION: 3.0,
		CNN_PARAMS: {
			MODEL_SUB_TYPE: 2,
		}
	})
	do_run(run_5c_3s)

	# besser
	run_5c_7s = cycle_base.copy()
	run_5c_7s.update({
		RUN_NAME_SUFFIX: "5c-7s-m2",
		CHUNK_DURATION: 7.0,
		CNN_PARAMS: {
			MODEL_SUB_TYPE: 2,
		}
	})
	do_run(run_5c_7s)

	run_6c_10s = cycle_base.copy()
	run_6c_10s.update({
		RUN_NAME_SUFFIX: "6c-10s-m3-l1l2",
		CHUNK_DURATION: 7.0,
		EPOCHS: 60,
		CNN_PARAMS: {
			MODEL_SUB_TYPE: 3,
			L1_REGULATION_WEIGHT: 0.0001,
			L2_REGULATION_WEIGHT: 0.0001,
			SCHEDULER: SCHEDULER_STEP,
		}
	})
	do_run(run_6c_10s)

	run_5c_5s_low = cycle_base.copy()
	run_5c_5s_low.update({
		RUN_NAME_SUFFIX: "5c-5s-low",
		CHUNK_DURATION: 5.0,
		CHUNK_HEARTCYCLE_COUNT: 5,
		CNN_PARAMS: {
			N_MELS: 64,
			HOP_LENGTH: 16,
			N_FFT: 512,
			MODEL_SUB_TYPE: 3,
		}
	})
	#do_run(run_5c_5s_low)

	run_8c_7s_m2 = cycle_base.copy()
	run_8c_7s_m2.update({
		RUN_NAME_SUFFIX: "8c_7s_m2",
		CHUNK_DURATION: 7.0,
		CHUNK_HEARTCYCLE_COUNT: 8,
		CNN_PARAMS: {
			MODEL_SUB_TYPE: 2,
		}
	})
	#do_run(run_8c_7s_m2)


	run_5c_7s_m3 = cycle_base.copy()
	run_5c_7s_m3.update({
		RUN_NAME_SUFFIX: "5c_5s_m3_long",
		CHUNK_DURATION: 5.0,
		CHUNK_HEARTCYCLE_COUNT: 7,
		EPOCHS: 100,
		LEARNING_RATE: 0.001,
		CNN_PARAMS: {
			MODEL_SUB_TYPE: 3,
			SCHEDULER: SCHEDULER_COSINE
		}
	})
	#do_run(run_5c_7s_m3)



	run_5c_7s_silu_kai = cycle_base.copy()
	run_5c_7s_silu_kai.update({
		RUN_NAME_SUFFIX: "5c-7s-silu-kai",
		CHUNK_DURATION: 7.0,
		CNN_PARAMS: {
			MODEL_SUB_TYPE: 3,
			ACTIVATION: ACTIVATION_SILU,
		}
	})
	#do_run(run_5c_7s_silu_kai)

	run_5c_7s_relu = cycle_base.copy()
	run_5c_7s_relu.update({
		RUN_NAME_SUFFIX: "5c-7s-relu",
		CHUNK_DURATION: 7.0,
		CNN_PARAMS: {
			MODEL_SUB_TYPE: 3,
			ACTIVATION: ACTIVATION_RELU,
		}
	})
	#do_run(run_5c_7s_relu)


	run_10c_5s_m2 = cycle_base.copy()
	run_10c_5s_m2.update({
		RUN_NAME_SUFFIX: "10c_5s_m2_long",
		CHUNK_DURATION: 5.0,
		CHUNK_HEARTCYCLE_COUNT: 10,
		EPOCHS: 80,
		LEARNING_RATE: 0.001,
		CNN_PARAMS: {
			L1_REGULATION_WEIGHT: 0.001,
			L2_REGULATION_WEIGHT: 0.001,
			OPTIMIZER: OPTIMIZER_SGD,
			SCHEDULER: SCHEDULER_STEP,
			MODEL_SUB_TYPE: 2,
		}
	})
	#do_run(run_10c_5s_m2)

	##################

	# Experiment 6: Adjust mel spectrogram parameters
	run6_dict = run_2022.copy()
	run6_dict.update({
		RUN_NAME_SUFFIX: "mel_adjust-half",
		CNN_PARAMS: {
			N_MELS: 256,
			HOP_LENGTH: 64,
			N_FFT: 512,
		}
	})


	# Experiment 8: Adjust learning rate and optimizer
	run8_dict = run_2022.copy()
	run8_dict.update({
		RUN_NAME_SUFFIX: "lr_optimizer",
		LEARNING_RATE: 0.001,
		CNN_PARAMS: {
			OPTIMIZER: OPTIMIZER_ADAM,
			SCHEDULER: SCHEDULER_COSINE,
		}
	})

	run10_dict = run_2022.copy()
	run10_dict.update({
		RUN_NAME_SUFFIX: "mel_adjust-double",
		CNN_PARAMS: {
			N_MELS: 1024,
			HOP_LENGTH: 256,
			N_FFT: 2048,
			}
	})


	# Experiment 11: long chunks
	run11_dict = run_2022.copy()
	run11_dict.update({
		RUN_NAME_SUFFIX: "long-chunks",
		CHUNK_DURATION: 15.0
	})

	# Experiment 12: Regularization
	run12_dict = run_2022.copy()
	run12_dict.update({
		RUN_NAME_SUFFIX: "l1regularization",
		CNN_PARAMS: {
			L1_REGULATION_WEIGHT: 0.0001,
		},
	})

	# Experiment 13: No filter
	run13_dict = run_2022.copy()
	run13_dict.update({
		RUN_NAME_SUFFIX: "l2regularization",
		CNN_PARAMS: {
			L2_REGULATION_WEIGHT: 0.0001,
		},
	})


	# Experiment 14: short chunks
	run14_dict = run_2022.copy()
	run14_dict.update({
		RUN_NAME_SUFFIX: "short-chunks",
		CHUNK_DURATION: 4.0
	})


	# Experiment 15: 
	run15_dict = run_2022.copy()
	run15_dict.update({
		RUN_NAME_SUFFIX: "crossentropy",
		CNN_PARAMS: {
			LOSS_FUNCTION: LOSS_CROSS_ENTROPY,
		}
	})


	# Experiment 16: 
	run16_dict = run_2022.copy()
	run16_dict.update({
		RUN_NAME_SUFFIX: "model2",
		CNN_PARAMS: {
			MODEL_SUB_TYPE: 2,
		}
	})


	# Experiment 17:
	run17_dict = run_2022.copy()
	run17_dict.update({
		RUN_NAME_SUFFIX: "model3",
		CNN_PARAMS: {
			MODEL_SUB_TYPE: 3,
		}
	})

	# Experiment 18:
	run18_dict = run_2022.copy()
	run18_dict.update({
		RUN_NAME_SUFFIX: "long-epoch-adam-cosine",
		EPOCHS: 80,
		LEARNING_RATE: 0.1,
		CNN_PARAMS: {
			OPTIMIZER: OPTIMIZER_ADAM,
			SCHEDULER: SCHEDULER_PLATEAU,
			MODEL_SUB_TYPE: 2,
		}
	})


	run19_dict = run_2022.copy()
	run19_dict.update({
		RUN_NAME_SUFFIX: "combined-optimized-2",
		EPOCHS: 80,
		LEARNING_RATE: 0.01,
		CHUNK_DURATION: 10.0,  # Längere Chunks für mehr Kontext
		CNN_PARAMS: {
			MODEL_SUB_TYPE: 2,  # Neues erweitertes Modell
			OPTIMIZER: OPTIMIZER_ADAM,
			SCHEDULER: SCHEDULER_STEP,
			N_MELS: 128,  # Erhöhte Mel-Spektrogramm-Auflösung
			HOP_LENGTH: 128,
			N_FFT: 1024,
			LOSS_FUNCTION: LOSS_FOCAL_LOSS,
			L2_REGULATION_WEIGHT: 0.0001,
			SIGNAL_FILTER: BUTTERPASS,
			BUTTERPASS_LOW: 20,
			BUTTERPASS_HIGH: 600,
			BUTTERPASS_ORDER: 3,
		}
	})

	###

	run_loop_test_dict = train_update_dict.copy()
	run_loop_test_dict.update({ \
		TRAIN_DATASET: PHYSIONET_2022, KFOLD_SPLITS: 1, EPOCHS: 2, METADATA_FRAC: 0.05})

	continue_test = {METADATA_FRAC: 0.05, TRAINING_CHECKPOINT: {EPOCH: 80, RUN_NAME: "2024-07-12_21-52-51_combined-optimized-2", FOLD: 1}, \
		LOAD_PREVIOUS_RUN_NAME: "2024-07-12_21-52-51_combined-optimized-2" ,RUN_NAME_SUFFIX: "continue-test", EPOCHS: 82}

	# do_run(run1_dict)
	# do_run(run2_dict)
	# do_run(run3_dict)
	# do_run(run4_dict)
	# do_run(run5_dict)

	# do_run(run6_dict)
	# do_run(run7_dict)
	# do_run(run8_dict)
	# do_run(run9_dict)
	# do_run(run10_dict)

	# do_run(run11_dict)
	# do_run(run12_dict)
	# do_run(run13_dict)
	# do_run(run14_dict)
	#do_run(run15_dict)
	#do_run(run17_dict)

	#do_run(run19_dict)
	# do_run(run16_dict)
	#do_run(run18_dict)

	# do_run(continue_test)

	# TODO test no downsampling of 2022
	do_run(run_loop_test_dict)

	plt.show()
