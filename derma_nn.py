import torch
import torch.nn as nn
import torch.nn.functional as F


class NNet(nn.Module):
    def __init__(self):
        super(NNet, self).__init__()
        self.fc1 = nn.Linear(2352, 128)  # 28*28*3=2352 (RGB instead of grayscale)
        self.fc2 = nn.Linear(128, 7)     # 7 classes instead of 10
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


"""
# conv1 output 28x28
# conv2 outputs 28x28
# pool and look for max in 2x2 patches
# pool output 14x14
# 14*14*64 = 12544
"""