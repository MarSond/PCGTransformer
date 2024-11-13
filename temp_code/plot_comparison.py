import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def create_model_comparison():
	# Daten vorbereiten
	models = [
		"2016 Fixed CNN", "2016 Fixed BEATs",
		"2022 Fixed CNN", "2022 Fixed BEATs", 
		"2022 Cycles CNN", "2022 Cycles BEATs"
	]

	# Metriken als Dictionary mit Werten und Standardabweichungen
	metrics_data = {
		"NMCC": [
			(0.865, 0.020), (0.856, 0.014),
			(0.736, 0.045), (0.661, 0.048),
			(0.716, 0.061), (0.681, 0.042)
		],
		"Precision": [
			(0.795, 0.070), (0.693, 0.044),
			(0.811, 0.169), (0.731, 0.118),
			(0.690, 0.161), (0.717, 0.131)
		],
		"Recall": [
			(0.789, 0.065), (0.888, 0.025),
			(0.392, 0.173), (0.202, 0.077),
			(0.411, 0.120), (0.272, 0.074)
		],
		"AUROC": [
			(0.961, 0.008), (0.938, 0.010),
			(0.795, 0.049), (0.657, 0.053),
			(0.754, 0.077), (0.701, 0.045)
		]
	}

	# Daten in DataFrame umwandeln
	data = []
	for metric, values in metrics_data.items():
		for i, (value, std) in enumerate(values):
			data.append({
				"Model": models[i],
				"Metric": metric,
				"Value": value,
				"Std": std
			})
	df = pd.DataFrame(data)

	# Plot erstellen
	with sns.axes_style("darkgrid"):
		fig, ax = plt.subplots(figsize=(18, 10))

		# Farbpalette ähnlich wie in den anderen Plots 
		colors = ["#1f77b4", "#2ca02c",  # 2016 Modelle - Blautöne
				"#ff7f0e", "#d62728",    # 2022 Fixed - Rottöne
				"#9467bd", "#8c564b"]    # 2022 Cycles - Violetttöne

		g = sns.barplot(
			data=df,
			x="Metric",
			y="Value",
			hue="Model",
			palette=colors,
			errorbar=None,  # Eigene Fehlerbalken
			width=0.8,
			ax=ax
		)

		# Manuelle Fehlerbalken und Beschriftungen
		num_metrics = len(metrics_data)
		num_models = len(models)
		width = 0.8 / num_models  # Breite der einzelnen Bars

		for i, metric in enumerate(metrics_data.keys()):
			for j, model in enumerate(models):
				mask = (df["Metric"] == metric) & (df["Model"] == model)
				value = df[mask]["Value"].iloc[0]
				std = df[mask]["Std"].iloc[0]

				# Position der Bar berechnen
				x_pos = i + (j - num_models/2 + 0.5) * width

				# Fehlerbalken
				ax.errorbar(x_pos, value, yerr=std, color="black", capsize=5,
							capthick=1.5, elinewidth=1.5, fmt="none")

				# Wert und Standardabweichung als Text
				ax.text(x_pos, value + std + 0.02,
						#f"{value:.2f}\n±{std:.3f}",
						f"{value:.2f}",
						ha="center", va="bottom",
						fontsize=13, fontweight="bold")

		# Titel und Labels mit einheitlicher Formatierung
		plt.title("Vergleich der Hauptmetriken über alle Modelle",
					fontsize=22, fontweight="bold", pad=20)
		plt.xlabel("Metrik", fontsize=18, fontweight="bold")
		plt.ylabel("Wert", fontsize=18, fontweight="bold")

		# Y-Achse anpassen
		plt.ylim(0, 1.1)  # Mehr Platz für Labels
		plt.yticks(np.arange(0, 1.1, 0.1),
					[f"{x:.1f}" for x in np.arange(0, 1.1, 0.1)],
					fontsize=14)

		# X-Achse
		plt.xticks(range(num_metrics), metrics_data.keys(),
					fontsize=14, fontweight="bold")

		plt.legend(bbox_to_anchor=(0.5, -0.1),
           loc="upper center",
           fontsize=15,
           title="Modelle",
           title_fontsize=16,
           frameon=True,
           ncol=6)  # Setze ncol auf die Anzahl der Einträge

		# Grid nur für y-Achse, im Hintergrund
		plt.grid(True, axis="y", linestyle="-", alpha=0.3, color="gray")
		plt.grid(axis="x", visible=False)
		ax.set_axisbelow(True)

		plt.tight_layout()

		return fig

# Plot erstellen und speichern
fig = create_model_comparison()
plt.savefig("model_comparison.png", dpi=300, bbox_inches="tight")
plt.close()
