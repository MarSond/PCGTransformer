import itertools
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import umap

from MLHelper.constants import *
from MLHelper.tools.utils import FileUtils, MLUtil, Plotting

# ruff: noqa: T201, E501

np.random.seed(SEED_VALUE)
pkl_path = Path("runs/2024-09-10_22-52-36_beats-test/other/embeddings.pkl")

with pkl_path.open("rb") as file:
	embeddings_data = pickle.load(file)

embeddings = embeddings_data["embeddings"]
labels = embeddings_data["labels"]


def create_umap_plot(embeddings, labels, n_neighbors, min_dist, is_3d=False):
	reducer = umap.UMAP(random_state=SEED_VALUE, n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, n_jobs=1)
	umap_embeddings = reducer.fit_transform(embeddings)


	fig = plt.figure(figsize=(10, 8))
	ax = fig.add_subplot(111)
	scatter = ax.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=labels, s=3, cmap="viridis")
	ax.set_title(f"2D UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})")

	plt.colorbar(scatter)
	return fig, ax

n_neighbors_range = [5, 15, 30, 50, 100]
min_dist_range = [0.0, 0.1, 0.25, 0.5, 0.8]


configs = list(itertools.product(n_neighbors_range, min_dist_range))


fig, axes = plt.subplots(len(n_neighbors_range), len(min_dist_range), figsize=(25, 25))
fig.suptitle("UMAP Grid Search Results", fontsize=16)

print("Plotting grid search results...")
for i, (n_neighbors, min_dist) in enumerate(configs):
	print(f"Start plotting for n={n_neighbors}, d={min_dist}")
	print(f"Plotting {i+1}/{len(configs)}")
	row = i // len(min_dist_range)
	col = i % len(min_dist_range)

	reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, n_jobs=6, low_memory=False)
	umap_embeddings = reducer.fit_transform(embeddings)

	scatter = axes[row, col].scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=labels, s=1, cmap="cividis")
	axes[row, col].set_title(f"n={n_neighbors}, d={min_dist}", fontsize=16)
	axes[row, col].set_xticks([])
	axes[row, col].set_yticks([])

plt.tight_layout()
plt.savefig("documents/umap_plots/umap_gridsearch.png", dpi=350, bbox_inches="tight")
print("Grid search plot saved.")
