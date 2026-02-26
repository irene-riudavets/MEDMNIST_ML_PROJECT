#!/usr/bin/env python

import torch
import torch.nn as nn
import medmnist
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import derma_nn as ch4nn

torch.manual_seed(42)

# LOAD TEST DATASET
test_dataset = medmnist.DermaMNIST(split='test', download=True, as_rgb=True)

# PREPROCESSING
def preprocess_test(dataset):
    images = torch.stack([torch.from_numpy(np.array(dataset[i][0])) for i in range(len(dataset))])
    labels = torch.tensor([dataset[i][1].item() for i in range(len(dataset))])
    
    images = images.float() / 255.0
    images = images.permute(0, 3, 1, 2)
    
    # Standardize images
    images = torch.stack([(e - e.mean()) / e.std() for e in images])
    return images, labels

test_images, test_labels = preprocess_test(test_dataset)

# LOAD TRAINED MODEL
trained_model_path = "output/cnn-dermamnist-trained.pth" 

model = ch4nn.CNNet()
model.load_state_dict(torch.load(trained_model_path))
model.eval()

# INFERENCE
print("Running test set")
with torch.no_grad():
    # We pass the images through the model
    # Note: If you have memory issues, you can do this in batches
    output = model(test_images)
    predictions = torch.argmax(output, dim=1)

# CALCULATE ACCURACY AND REPORT
acc = torch.sum(predictions == test_labels).item() / len(test_labels)

print("\n" + "="*30)
print(f"TEST SET ACCURACY: {acc:.4f}")
print("="*30)

print("\nFull Classification Report (Test Set):")
print(classification_report(test_labels.numpy(), predictions.numpy()))
