from pathlib import Path

import numpy as np

from MLHelper import constants as const
from MLHelper.metrics import metrics

target_list = [
	"2024-09-18_23-40-18_2016_fixed_beats_knn_finalrun",
	"2024-09-19_01-01-49_2016_fixed_cnn_finalrun",
	"2024-09-21_12-19-45_2022_cycles_beats_knn_finalrun_v2",
	"2024-09-22_01-01-39_2022_cycles_cnn_finalrun_v2",
	"2024-09-22_15-36-53_2022_fixed_cnn_finalrun_v2",
	"2024-09-23_09-11-26_2022_fixed_beats_knn_finalrun_v5"
]

def calculate_f2_score(precision, recall):
	return 5 * (precision * recall) / (4 * precision + recall)

def calculate_f05_score(precision, recall):
	return 1.25 * (precision * recall) / (0.25 * precision + recall)

def add_f2_f05_scores_to_metrics(metric_data):
	for mode in [const.TRAINING, const.VALIDATION]:
		if mode in metric_data[const.FOLD_DATA]:
			for fold in metric_data[const.FOLD_DATA][mode]:
				for epoch_data in fold:
					precision = epoch_data[const.METRICS_PRECISION]
					recall = epoch_data[const.METRICS_RECALL]
					if const.METRICS_F2 not in epoch_data:
						epoch_data[const.METRICS_F2] = calculate_f2_score(precision, recall)
					if const.METRICS_F05 not in epoch_data:
						epoch_data[const.METRICS_F05] = calculate_f05_score(precision, recall)

	# Aktualisiere die Durchschnittswerte
	for mode in [const.TRAINING, const.VALIDATION]:
		if mode in metric_data[const.AVERAGES]:
			avg_data = metric_data[const.AVERAGES][mode]
			if len(avg_data) == 0:
				continue
			precision_data = avg_data[const.METRICS_PRECISION]
			recall_data = avg_data[const.METRICS_RECALL]

			for metric_name, calc_func in [(const.METRICS_F2, calculate_f2_score),
											(const.METRICS_F05, calculate_f05_score)]:
				if metric_name not in avg_data:
					metric_scores = []
					for epoch_index, (p, r) in enumerate(zip(precision_data, recall_data)):
						all_fold_scores = calc_func(
							np.array([fold[epoch_index][const.METRICS_PRECISION] for fold in metric_data[const.FOLD_DATA][mode]]),
							np.array([fold[epoch_index][const.METRICS_RECALL] for fold in metric_data[const.FOLD_DATA][mode]])
						)
						mean_score = calc_func(p[const.MEAN], r[const.MEAN])
						std_score = np.std(all_fold_scores)
						metric_scores.append({
							const.MEAN: mean_score,
							const.STD: std_score,
							const.EPOCH: p[const.EPOCH]
						})
					avg_data[metric_name] = metric_scores

	return metric_data

for target_name in target_list:
	run_target_folder = Path("final_runs") / f"fix_{target_name}"
	run_target_folder.mkdir(parents=True, exist_ok=True)
	fake_config = {
		const.RUN_RESULTS_PATH: run_target_folder,
	}

	run = Path("final_runs") / target_name

	metric = metrics.MetricsTracker(config=fake_config)

	metric.load_metrics_state(Path(run) / const.METRICS_FOLDER / const.FILENAME_METRICS_VALUE)

	# Füge F2-Score hinzu oder berechne ihn, falls er fehlt
	metric.all_data = add_f2_f05_scores_to_metrics(metric.all_data)

	# Aktualisiere die internen Datenstrukturen des MetricsTracker-Objekts
	metric.all_metrics = metric.all_data[const.FOLD_DATA]
	metric.train_metrics.averages = metric.all_data[const.AVERAGES][const.TRAINING]
	metric.valid_metrics.averages = metric.all_data[const.AVERAGES][const.VALIDATION]

	metric.save_end_results_txt()
	metric.save_metrics()

	metric.plot_average_roc_pr_curves()
	metric.create_summary_plot()
	metric.plot_all_metrics()
