#!/usr/bin/env python
"""
Complete Machine Learning Analysis of DermaMNIST Dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
import medmnist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import seaborn as sns

import derma_nn as ch4nn

torch.manual_seed(42)
np.random.seed(42)

print("="*60)
print("DERMAMNIST ML PROJECT - COMPLETE ANALYSIS")
print("="*60)

# LOAD DATASETS

train_dataset = medmnist.DermaMNIST(split='train', download=True, as_rgb=True)
val_dataset   = medmnist.DermaMNIST(split='val', download=True, as_rgb=True)
test_dataset  = medmnist.DermaMNIST(split='test', download=True, as_rgb=True)

print(f"Training set: {len(train_dataset)} samples")
print(f"Validation set: {len(val_dataset)} samples")
print(f"Test set: {len(test_dataset)} samples")

# DATA PREPROCESSING

def preprocess_dataset(dataset):
    images = torch.stack([torch.from_numpy(np.array(dataset[i][0])) for i in range(len(dataset))])
    labels = torch.tensor([dataset[i][1].item() for i in range(len(dataset))])
    images = images.float() / 255.0
    images = images.reshape(len(images), 2352)
    images = torch.stack([(e - e.mean()) / e.std() for e in images])
    return images, labels

train_images, train_labels  = preprocess_dataset(train_dataset)
val_images, val_labels      = preprocess_dataset(val_dataset)
test_images, test_labels    = preprocess_dataset(test_dataset)

# MODEL TRAINING

model = ch4nn.NNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.NLLLoss()

batchsize = 128
nbatches = len(train_images) // batchsize + 1
train_losses = []

for epoch in range(20): #Number of epochs (how many time the model "sees" all the data)
    epoch_losses = []
    for i in range(nbatches):
        i1, i2 = i * batchsize, min((i + 1) * batchsize, len(train_images))
        if i1 >= i2:
            break

        optimizer.zero_grad()
        out = model(train_images[i1:i2])
        loss = loss_fn(out, train_labels[i1:i2])
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())

    train_losses.append(np.mean(epoch_losses))
    print(f"Epoch {epoch+1}/3 - Average Loss: {train_losses[-1]:.3f}")

torch.save(model.state_dict(), "output/20-TRAIN-nn-dermamnist-trained.pth")

# """"""""""""""""
# VALIDATION
# """"""""""""""""

# MODEL EVALUATION

model.eval()
with torch.no_grad():
    train_acc = accuracy_score(
        train_labels.numpy(),
        model(train_images).argmax(dim=1).numpy()
    )
    val_pred = model(val_images).argmax(dim=1)
    val_acc = accuracy_score(val_labels.numpy(), val_pred.numpy())

print(f"\n   Training Accuracy: {train_acc:.3f}")
print(f"   Validation Accuracy: {val_acc:.3f}")

# CONFUSION MATRIX
cm = confusion_matrix(val_labels.numpy(), val_pred.numpy())

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Validation Set')
plt.tight_layout()
plt.savefig('output/20-VAL-confusion_matrix.png', dpi=150)
plt.close()

# CLASSIFICATION REPORT
print("\n8. Classification Report (Validation Set):")
print(classification_report(val_labels.numpy(), val_pred.numpy(), zero_division=0))

# TRAINING LOSS PLOT
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(train_losses)+1), train_losses, 'bo-')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.title('Training Loss')
plt.tight_layout()
plt.savefig('output/20-training_loss.png', dpi=150)
plt.close()
