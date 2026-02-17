#!/usr/bin/env python

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import medmnist
import numpy as np

# Load all datasets
train = medmnist.DermaMNIST(split='train', transform=None, download=True, as_rgb=True)
val = medmnist.DermaMNIST(split='val', transform=None, download=True, as_rgb=True)
test = medmnist.DermaMNIST(split='test', transform=None, download=True, as_rgb=True)

# Combine datasets
images = np.array([np.array(train[i][0]) for i in range(len(train))] + 
                  [np.array(val[i][0]) for i in range(len(val))] + 
                  [np.array(test[i][0]) for i in range(len(test))])
labels = np.array([train[i][1].item() for i in range(len(train))] + 
                  [val[i][1].item() for i in range(len(val))] + 
                  [test[i][1].item() for i in range(len(test))])

print(images.shape)
images = images.astype(float) / 255.0
fim = images.reshape(len(images), 28*28*3)
fims = [(e-e.mean())/e.std() for e in fim]
pca2 = PCA(n_components=50)
xpca2s = pca2.fit_transform(fims) # Use only the first 5000 samples for t-SNE and KMeans to speed up computation (fims[0:5000])

tsne = TSNE(n_components=2, perplexity=30, random_state=7, init='pca')
Xtsne = tsne.fit_transform(xpca2s)
km = KMeans(n_clusters=7, random_state=7).fit(xpca2s)

plt.scatter(Xtsne[:, 0], Xtsne[:, 1],
        c=km.labels_, cmap='tab10', s=5, alpha=0.7)
plt.colorbar(label='Cluster Label')
plt.savefig("M5-HPC/MEDMNIST_ML_PROJECT/output/kmeans-tsne-dermamnist-30-full.png")
plt.show()