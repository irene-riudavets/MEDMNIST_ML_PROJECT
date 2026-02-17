#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.optim as optim
import medmnist
import numpy as np

import derma_nn as ch4nn

torch.manual_seed(42)

# Load DermaMNIST training set
train_dataset = medmnist.DermaMNIST(split='train', transform=None, download=True, as_rgb=True)

images = torch.stack([torch.from_numpy(np.array(train_dataset[i][0])) for i in range(len(train_dataset))])
labels = torch.tensor([train_dataset[i][1].item() for i in range(len(train_dataset))])

images = images.float() / 255.0
images = images.reshape(len(images), 2352)          # 28*28*3=2352 (RGB)
images = torch.stack([(e-e.mean())/e.std() for e in images])

nimgs = images.shape[0]
batchsize = 128
nbatches = nimgs // 128 + 1

trained_model_path = "M5-HPC/MEDMNIST_ML_PROJECT/output/nn-dermamnist-trained.pth"

model     = ch4nn.NNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
nll_loss  = nn.NLLLoss()

for i in range(nbatches):
    i1 = i * batchsize
    i2 = min((i+1) * batchsize, nimgs)
    imgbatch = images[i1:i2, :]
    optimizer.zero_grad()
    out  = model.forward(imgbatch)
    loss = nll_loss(out, labels[i1:i2])
    print("{}:{:.3f}".format(i, loss))
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), trained_model_path)