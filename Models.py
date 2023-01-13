import torch.nn.functional as F
import torch.nn as nn
import torch
import torchvision

import torch
import torch.nn as nn
import math
import torch.nn.functional as F



class Identity(nn.Module):
  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x

class MyResnet101_(nn.Module):

  def __init__(self, n_classes):
    super(MyResnet101_, self).__init__()
    self.model = torchvision.models.resnet101()
    self.model.fc = nn.Linear(in_features=2048, out_features=n_classes)
    self.model.tanh = nn.Tanh()

  def forward(self, x):
    out = self.model.forward(x)
    # out = self.SM(out)
    out = self.tanh(out)
    return  out

class Net(nn.Module):
  def __init__(self, n_channels, n_classes):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(n_channels, 6, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(16 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, n_classes)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 16 * 5 * 5)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x
