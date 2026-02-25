#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import medmnist
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import (
    confusion_matrix,
    adjusted_rand_score,
    normalized_mutual_info_score
)

# -----------------------------
# 1. Load DermaMNIST
# -----------------------------
train = medmnist.DermaMNIST(split='train', download=True, as_rgb=True)
val   = medmnist.DermaMNIST(split='val',   download=True, as_rgb=True)
test  = medmnist.DermaMNIST(split='test',  download=True, as_rgb=True)

images = np.array(
    [np.array(train[i][0]) for i in range(len(train))] +
    [np.array(val[i][0])   for i in range(len(val))] +
    [np.array(test[i][0])  for i in range(len(test))]
)

labels = np.array(
    [train[i][1].item() for i in range(len(train))] +
    [val[i][1].item()   for i in range(len(val))] +
    [test[i][1].item()  for i in range(len(test))]
)

print("Images shape:", images.shape)
print("Unique labels:", np.unique(labels))

# -----------------------------
# 2. Normalize & flatten
# -----------------------------
images = images.astype(np.float32) / 255.0
N = images.shape[0]
fim = images.reshape(N, -1)

# Per-image standardization
mean = fim.mean(axis=1, keepdims=True)
std  = fim.std(axis=1, keepdims=True)
std[std == 0] = 1.0
fims = (fim - mean) / std

# -----------------------------
# 3. Subsample for t-SNE
# -----------------------------
n_tsne = 5000
fims_sub = fims[:n_tsne]
labels_sub = labels[:n_tsne]

# -----------------------------
# 4. PCA (for clustering)
# -----------------------------
pca = PCA(n_components=50, random_state=42)
xpca = pca.fit_transform(fims_sub)

# -----------------------------
# 5. KMeans clustering
# -----------------------------
n_clusters = 7
kmeans = KMeans(n_clusters=n_clusters, random_state=7)
clusters = kmeans.fit_predict(xpca)

# -----------------------------
# 6. Quantitative validation
# -----------------------------
ari = adjusted_rand_score(labels_sub, clusters)
nmi = normalized_mutual_info_score(labels_sub, clusters)

print(f"\nAdjusted Rand Index (ARI): {ari:.3f}")
print(f"Normalized Mutual Information (NMI): {nmi:.3f}")

# Confusion matrix
cm = confusion_matrix(labels_sub, clusters)
cm_df = pd.DataFrame(
    cm,
    index=[f"True_{i}" for i in range(cm.shape[0])],
    columns=[f"Cluster_{i}" for i in range(cm.shape[1])]
)

print("\nConfusion matrix (rows=true labels, cols=clusters):")
print(cm_df)

# -----------------------------
# 7. Majority-vote label mapping
# -----------------------------
cluster_label_map = {}

for c in np.unique(clusters):
    true_labels_in_cluster = labels_sub[clusters == c]
    majority_label = np.bincount(true_labels_in_cluster).argmax()
    cluster_label_map[c] = majority_label

print("\nCluster → majority true label mapping:")
for k, v in cluster_label_map.items():
    print(f"Cluster {k} → Label {v}")

# -----------------------------
# 8. t-SNE visualization
# -----------------------------
tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=7,
    init='pca'
)
Xtsne = tsne.fit_transform(xpca)

# Side-by-side plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(
    Xtsne[:, 0], Xtsne[:, 1],
    c=labels_sub, cmap='tab10', s=5
)
axes[0].set_title("True DermaMNIST labels")
axes[0].set_xlabel("t-SNE 1")
axes[0].set_ylabel("t-SNE 2")

axes[1].scatter(
    Xtsne[:, 0], Xtsne[:, 1],
    c=clusters, cmap='tab10', s=5
)
axes[1].set_title("KMeans clusters")
axes[1].set_xlabel("t-SNE 1")
axes[1].set_ylabel("t-SNE 2")

plt.tight_layout()
plt.savefig(
    "output/tsne-true-vs-kmeans.png",
    dpi=300
)
plt.show()
