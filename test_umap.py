import itertools
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import umap

from MLHelper.constants import *
from MLHelper.tools.utils import FileUtils, MLUtil, Plotting

np.random.seed(SEED_VALUE)

def create_umap_plots(run_name, n_neighbors_range, min_dist_range):
	pkl_path = Path(f"final_runs/{run_name}/other/fold1_embeddings.pkl")
	output_dir = Path(f"final_runs/{run_name}/other")

	with pkl_path.open("rb") as file:
		embeddings_data = pickle.load(file)

	embeddings = embeddings_data["embeddings"] #[:int(len(embeddings_data["embeddings"]) * 0.1)]
	labels = embeddings_data["labels"] #[:int(len(embeddings_data["embeddings"]) * 0.1)]

	configs = list(itertools.product(n_neighbors_range, min_dist_range))

	for supervised in [True, False]:
		fig, axes = plt.subplots(len(n_neighbors_range), len(min_dist_range), figsize=(25, 25))
		fig.suptitle(f"UMAP Grid Search Results ({'Supervised' if supervised else 'Unsupervised'}), n_components=2", fontsize=20, weight="bold")

		for i, (n_neighbors, min_dist) in enumerate(configs):
			row = i // len(min_dist_range)
			col = i % len(min_dist_range)

			reducer = umap.UMAP(random_state=SEED_VALUE, n_neighbors=n_neighbors, min_dist=min_dist, \
				n_components=2, n_jobs=6, low_memory=False)
			umap_embeddings = reducer.fit_transform(embeddings, y=labels if supervised else None)

			ax = axes[row, col]
			ax.clear()
			ax.set_facecolor("#f0f0f0")

			if labels is not None:
				unique_labels = np.unique(labels)
				colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))
				for j, unique_label in enumerate(unique_labels):
					indices = np.where(labels == unique_label)
					ax.scatter(umap_embeddings[indices, 0], umap_embeddings[indices, 1], \
						c=[colors[j]], s=2, label=f"class {str(int(unique_label))}")
				ax.legend(loc="upper right", markerscale=5)
			else:
				ax.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], s=2)

			ax.set_title(f"n_neighbors={n_neighbors}, min_dist={min_dist}", fontsize=15, weight="bold")
			ax.set_xlabel("dim 1", fontsize=14)
			ax.set_ylabel("dim 2", fontsize=14)
			ax.tick_params(labelsize=12)
			print(f"UMAP 2D plot created for n={n_neighbors}, d={min_dist} in run {run_name}")

		plt.tight_layout(rect=[0.07, 0.05, 1, 0.95])
		filename = f"umap_gridsearch_{'supervised' if supervised else 'unsupervised'}.png"
		plt.savefig(output_dir / filename, dpi=350, bbox_inches="tight")
		plt.close(fig)

# Beispielaufruf für jeden Run
run_names = [
	"2024-09-18_23-40-18_2016_fixed_beats_knn_finalrun",
	"2024-09-21_12-19-45_2022_cycles_beats_knn_finalrun_v2",
	"2024-10-09_21-21-06_2022_fixed_beats_knn_finalrun_v15"
]

n_neighbors_range = [5, 10, 15, 30, 50]
min_dist_range = [0.0, 0.1, 0.25, 0.5, 0.8]

for run_name in run_names:
	create_umap_plots(run_name, n_neighbors_range, min_dist_range)
	print(f"UMAP plots created for {run_name}")
