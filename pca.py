#!/usr/bin/env python

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
print(f"Unique labels: {np.unique(labels)}")  # Check what labels we have
images = images.astype(float) / 255.0
fim = images.reshape(len(images), 28*28*3)
fims = [(e-e.mean())/e.std() for e in fim]
pca = PCA(n_components=2, random_state=42)
xpca = pca.fit_transform(fims)
scatter = plt.scatter(xpca[:,0], xpca[:,1], c=labels, cmap='tab10', alpha=0.6, s=10)
legend1 = plt.legend(*scatter.legend_elements(),
                     title="DermaMNIST classes", loc='best',
                     bbox_to_anchor=(1, 1))
plt.gca().add_artist(legend1)
plt.tight_layout()  # This should fix the legend getting cut off
plt.savefig("M5-HPC/MEDMNIST_ML_PROJECT/output/pca-dermamnist.png", bbox_inches='tight')  # Also use bbox_inches='tight'
plt.show()