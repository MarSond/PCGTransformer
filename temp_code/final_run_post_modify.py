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
	"2024-10-09_21-21-06_2022_fixed_beats_knn_finalrun_v15"
]

def calculate_fbeta_score(precision, recall, beta):
	if precision + recall == 0:
		return 0.0
	beta_squared = beta ** 2
	return (1 + beta_squared) * (precision * recall) / (beta_squared * precision + recall)

def add_f2_f05_scores_to_metrics(metric_data):
	for mode in [const.TRAINING, const.VALIDATION]:
		if mode in metric_data[const.FOLD_DATA]:
			for fold in metric_data[const.FOLD_DATA][mode]:
				for epoch_data in fold:
					precision = epoch_data[const.METRICS_PRECISION]
					recall = epoch_data[const.METRICS_RECALL]
					if const.METRICS_F2 not in epoch_data:
						epoch_data[const.METRICS_F2] = calculate_fbeta_score(precision, recall, beta=2)
					if const.METRICS_F05 not in epoch_data:
						epoch_data[const.METRICS_F05] = calculate_fbeta_score(precision, recall, beta=0.5)

	# Aktualisiere die Durchschnittswerte
	for mode in [const.TRAINING, const.VALIDATION]:
		if mode in metric_data[const.AVERAGES]:
			avg_data = metric_data[const.AVERAGES][mode]
			if len(avg_data) == 0:
				continue
			precision_data = avg_data[const.METRICS_PRECISION]
			recall_data = avg_data[const.METRICS_RECALL]

			for metric_name, beta  in [(const.METRICS_F2, 2),
											(const.METRICS_F05, 0.5)]:
				if metric_name not in avg_data:
					metric_scores = []
					for epoch_index, (p, r) in enumerate(zip(precision_data, recall_data)):
						all_fold_scores = np.array([
							calculate_fbeta_score(
								fold[epoch_index][const.METRICS_PRECISION],
								fold[epoch_index][const.METRICS_RECALL],
								beta
							) for fold in metric_data[const.FOLD_DATA][mode]
						])
						mean_score = calculate_fbeta_score(p[const.MEAN], r[const.MEAN], beta)
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

	metric.plot_average_cm()
	metric.plot_average_roc_pr_curves()
	metric.create_summary_plot()
	metric.plot_all_metrics()
