#!/usr/bin/env python

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import medmnist
import numpy as np

# Load datasets
train = medmnist.DermaMNIST(split='train', download=True, as_rgb=True)
val   = medmnist.DermaMNIST(split='val',   download=True, as_rgb=True)
test  = medmnist.DermaMNIST(split='test',  download=True, as_rgb=True)

# Combine datasets
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

print(images.shape)

# Normalize
images = images.astype(np.float32) / 255.0
N = images.shape[0]

# Flatten
fim = images.reshape(N, -1)

# Standardize per image
mean = fim.mean(axis=1, keepdims=True)
std  = fim.std(axis=1, keepdims=True)
std[std == 0] = 1.0
fims = (fim - mean) / std

# ---- Subsample for t-SNE (VERY IMPORTANT) ----
n_tsne = 5000
fims_sub = fims[:n_tsne]
labels_sub = labels[:n_tsne]

# PCA (for both t-SNE and KMeans)
pca = PCA(n_components=50, random_state=42)
xpca = pca.fit_transform(fims_sub)

# KMeans in PCA space
km = KMeans(n_clusters=7, random_state=7)
clusters = km.fit_predict(xpca)

# t-SNE for visualization only
tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=7,
    init='pca'
)
Xtsne = tsne.fit_transform(xpca)

# Plot
plt.figure(figsize=(7, 6))
plt.scatter(
    Xtsne[:, 0], Xtsne[:, 1],
    c=clusters, cmap='tab10', s=6, alpha=0.7
)

plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.colorbar(label="KMeans cluster")
plt.tight_layout()
plt.savefig(
    "output/tsne_corrected_dermamnist-30.png"
)
plt.show()
