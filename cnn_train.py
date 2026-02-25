#!/usr/bin/env python
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
from sklearn.utils.class_weight import compute_class_weight  # Added for balancing
import seaborn as sns
import derma_nn as ch4nn

torch.manual_seed(42)
np.random.seed(42)

# Load Datasets
train_dataset = medmnist.DermaMNIST(split='train', download=True, as_rgb=True)
val_dataset   = medmnist.DermaMNIST(split='val', download=True, as_rgb=True)

# DATA PREPROCESSING (Modified for CNN: keeps 28x28x3 shape)
def preprocess_dataset(dataset):
    # Stack images and permute to (N, Channels, H, W) for CNN
    images = torch.stack([torch.from_numpy(np.array(dataset[i][0])) for i in range(len(dataset))])
    labels = torch.tensor([dataset[i][1].item() for i in range(len(dataset))])
    images = images.float() / 255.0
    images = images.permute(0, 3, 1, 2) # CNNs need (N, C, H, W)
    images = torch.stack([(e - e.mean()) / e.std() for e in images])
    return images, labels

train_images, train_labels = preprocess_dataset(train_dataset)
val_images, val_labels     = preprocess_dataset(val_dataset)

# MODEL TRAINING

model = ch4nn.CNNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# CLASS BALANCING: Calculate weights
weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels.numpy())
class_weights = torch.tensor(weights, dtype=torch.float)
loss_fn = nn.NLLLoss(weight=class_weights) # Apply weights here

batchsize = 128
nbatches = len(train_images) // batchsize + 1
train_losses = []

epochs = 20 # Epoch --> how many times the model "sees" the data
for epoch in range(epochs):
    epoch_losses = []
    for i in range(nbatches):
        i1, i2 = i * batchsize, min((i + 1) * batchsize, len(train_images))
        if i1 >= i2: break

        optimizer.zero_grad()
        out = model(train_images[i1:i2]) # Images are already (N, 3, 28, 28)
        loss = loss_fn(out, train_labels[i1:i2])
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())

    train_losses.append(np.mean(epoch_losses))
    print(f"Epoch {epoch+1}/{epochs} - Loss: {train_losses[-1]:.3f}")

torch.save(model.state_dict(), "output/cnn-dermamnist-trained.pth")


# MODEL EVALUATION (VALIDATION ONLY)

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

# 7. CONFUSION MATRIX (VALIDATION)

cm = confusion_matrix(val_labels.numpy(), val_pred.numpy())
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Validation Set')
plt.tight_layout()
plt.savefig('output/cnn_confusion_matrix.png', dpi=150)
plt.close()


# CLASSIFICATION REPORT (VALIDATION)

print(classification_report(val_labels.numpy(), val_pred.numpy()))


# TRAINING LOSS PLOT

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(train_losses)+1), train_losses, 'bo-')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.title('Training Loss')
plt.tight_layout()
plt.savefig('output/cnn_training_loss.png', dpi=150)
plt.close()
