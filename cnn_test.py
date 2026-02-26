#!/usr/bin/env python

import torch
import torch.nn as nn
import medmnist
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import derma_nn as ch4nn

torch.manual_seed(42)

# 1. LOAD TEST DATASET
test_dataset = medmnist.DermaMNIST(split='test', download=True, as_rgb=True)

# 2. PREPROCESSING
def preprocess_test(dataset):
    # Stack images into a tensor
    images = torch.stack([torch.from_numpy(np.array(dataset[i][0])) for i in range(len(dataset))])
    labels = torch.tensor([dataset[i][1].item() for i in range(len(dataset))])
    
    images = images.float() / 255.0
    # Move channels to (N, C, H, W) for the CNN
    images = images.permute(0, 3, 1, 2)
    
    # Standardize each image
    images = torch.stack([(e - e.mean()) / e.std() for e in images])
    return images, labels

test_images, test_labels = preprocess_test(test_dataset)

# 3. LOAD TRAINED MODEL
trained_model_path = "output/cnn-dermamnist-trained.pth" 

model = ch4nn.CNNet()
model.load_state_dict(torch.load(trained_model_path))
model.eval()

# 4. INFERENCE
print("Running inference on test set...")
with torch.no_grad():
    # We pass the images through the model
    # Note: If you have memory issues, you can do this in batches
    output = model(test_images)
    predictions = torch.argmax(output, dim=1)

# 5. CALCULATE ACCURACY AND REPORT
acc = torch.sum(predictions == test_labels).item() / len(test_labels)

print("\n" + "="*30)
print(f"TEST SET ACCURACY: {acc:.4f}")
print("="*30)

print("\nFull Classification Report (Test Set):")
print(classification_report(test_labels.numpy(), predictions.numpy()))
