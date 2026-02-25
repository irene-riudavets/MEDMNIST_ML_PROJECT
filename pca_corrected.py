#!/usr/bin/env python

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
print(f"Unique labels: {np.unique(labels)}")

# Normalize to [0,1]
images = images.astype(np.float32) / 255.0

# Flatten
N = images.shape[0]
fim = images.reshape(N, -1)

# Per-image standardization (MNIST-style)
mean = fim.mean(axis=1, keepdims=True)
std  = fim.std(axis=1, keepdims=True)
std[std == 0] = 1.0  # safety
fims = (fim - mean) / std

# PCA
pca = PCA(n_components=2, random_state=42)
xpca = pca.fit_transform(fims)

# Plot
plt.figure(figsize=(7, 6))
scatter = plt.scatter(
    xpca[:, 0], xpca[:, 1],
    c=labels, cmap='tab10', alpha=0.6, s=10
)

legend1 = plt.legend(
    *scatter.legend_elements(),
    title="DermaMNIST classes",
    loc='best',
    bbox_to_anchor=(1, 1)
)
plt.gca().add_artist(legend1)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig(
    "output/pca_corrected_dermamnist.png",
    bbox_inches='tight'
)
plt.show()
