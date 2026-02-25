#!/usr/bin/env python

"""
Script to display a 3x3 grid of DermaMNIST images
"""

import torch
import matplotlib.pyplot as plt
import medmnist
from medmnist import INFO
import numpy as np

# Specify the dataset name
dataset_name = 'dermamnist'

# Load the train dataset without any transforms initially
# We'll normalize manually like in the MNIST example
train_dataset = medmnist.DermaMNIST(
    split='train',
    download=True,
    as_rgb=True
)

print(f"Dataset loaded. Total images: {len(train_dataset)}")

# Get the first 9 images and labels
images = []
labels = []
for i in range(9):
    img, label = train_dataset[i]
    images.append(img)
    labels.append(label)

print(f"Image shape (before processing): {images[0].size}")  # PIL Image size

# Convert PIL images to numpy arrays, then to tensors
images_tensor = torch.stack([torch.from_numpy(np.array(img)) for img in images])
print(f"Tensor shape: {images_tensor.shape}")  # Should be [9, 28, 28, 3]

# Convert to float and normalize to [0, 1]
images_tensor = images_tensor.float() / 255.0

# Normalize each image individually (subtract mean, divide by std)
# This is equivalent to: images = [(e-e.mean())/e.std() for e in images]
normalized_images = []
for img in images_tensor:
    img_normalized = (img - img.mean()) / img.std()
    normalized_images.append(img_normalized)

# Stack back into a tensor
normalized_images = torch.stack(normalized_images)

# Create a 3x3 grid
fig, axes = plt.subplots(3, 3, figsize=(5, 5))
for i, ax in enumerate(axes.flat):
    # Display the image
    ax.imshow(normalized_images[i])
    ax.axis('off')
    # Optionally add the label as title
    # ax.set_title(f'{labels[i].item()}', fontsize=8)

plt.tight_layout()
plt.savefig("M5-HPC/MEDMNIST_ML_PROJECT/final_version/output/dermamnist-grid.png", dpi=300)
plt.show()