import torch
import torch.nn as nn
import torch.nn.functional as F

""" Neural Network """

class NNet(nn.Module):
    def __init__(self):
        super(NNet, self).__init__()
        self.fc1 = nn.Linear(2352, 128)  # 28*28*3=2352 (x3 because it is RGB and not grayscale)
        self.fc2 = nn.Linear(128, 7)     # 7 classes
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


"""Convolutional Neural Network """

class CNNet(nn.Module):
    def __init__(self):
        super(CNNet, self).__init__()
        # Convolutional layers preserve spatial structure
        self.conv1 = nn.Conv2d(3, 32, 3, 1, padding=1)   # 3 input channels (RGB)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, padding=1) # Extra conv layer
        self.pool = nn.MaxPool2d(2, 2)

        # After 2 pooling: 28 -> 14 -> 7
        # 7*7*128 = 6272
        self.fc1 = nn.Linear(6272, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 7)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Input is already [batch, 3, 28, 28]
        x = F.relu(self.conv1(x))
        x = self.pool(x)              # 28 -> 14
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)              # 14 -> 7
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output

