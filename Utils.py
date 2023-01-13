"""
Created on 23/10/2022

@author: Dimitrios Papaioannou
"""


import torch
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from models.resnet_ova import *
from Dataset import *
import os
import os.path as osp
import torch.backends.cudnn as cudnn
import argparse
# Transformation of 1x10 to 45x10 for Multiclass vs Binary
# M = np.array(
#     [
#         [1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
#         [1, 0, -1, 0, 0, 0, 0, 0, 0, 0],
#         [1, 0, 0, -1, 0, 0, 0, 0, 0, 0],
#         [1, 0, 0, 0, -1, 0, 0, 0, 0, 0],
#         [1, 0, 0, 0, 0, -1, 0, 0, 0, 0],
#         [1, 0, 0, 0, 0, 0, -1, 0, 0, 0],
#         [1, 0, 0, 0, 0, 0, 0, -1, 0, 0],
#         [1, 0, 0, 0, 0, 0, 0, 0, -1, 0],
#         [1, 0, 0, 0, 0, 0, 0, 0, 0, -1],
#         [0, 1, -1, 0, 0, 0, 0, 0, 0, 0],
#         [0, 1, 0, -1, 0, 0, 0, 0, 0, 0],
#         [0, 1, 0, 0, -1, 0, 0, 0, 0, 0],
#         [0, 1, 0, 0, 0, -1, 0, 0, 0, 0],
#         [0, 1, 0, 0, 0, 0, -1, 0, 0, 0],
#         [0, 1, 0, 0, 0, 0, 0, -1, 0, 0],
#         [0, 1, 0, 0, 0, 0, 0, 0, -1, 0],
#         [0, 1, 0, 0, 0, 0, 0, 0, 0, -1],
#         [0, 0, 1, -1, 0, 0, 0, 0, 0, 0],
#         [0, 0, 1, 0, -1, 0, 0, 0, 0, 0],
#         [0, 0, 1, 0, 0, -1, 0, 0, 0, 0],
#         [0, 0, 1, 0, 0, 0, -1, 0, 0, 0],
#         [0, 0, 1, 0, 0, 0, 0, -1, 0, 0],
#         [0, 0, 1, 0, 0, 0, 0, 0, -1, 0],
#         [0, 0, 1, 0, 0, 0, 0, 0, 0, -1],
#         [0, 0, 0, 1, -1, 0, 0, 0, 0, 0],
#         [0, 0, 0, 1, 0, -1, 0, 0, 0, 0],
#         [0, 0, 0, 1, 0, 0, -1, 0, 0, 0],
#         [0, 0, 0, 1, 0, 0, 0, -1, 0, 0],
#         [0, 0, 0, 1, 0, 0, 0, 0, -1, 0],
#         [0, 0, 0, 1, 0, 0, 0, 0, 0, -1],
#         [0, 0, 0, 0, 1, -1, 0, 0, 0, 0],
#         [0, 0, 0, 0, 1, 0, -1, 0, 0, 0],
#         [0, 0, 0, 0, 1, 0, 0, -1, 0, 0],
#         [0, 0, 0, 0, 1, 0, 0, 0, -1, 0],
#         [0, 0, 0, 0, 1, 0, 0, 0, 0, -1],
#         [0, 0, 0, 0, 0, 1, -1, 0, 0, 0],
#         [0, 0, 0, 0, 0, 1, 0, -1, 0, 0],
#         [0, 0, 0, 0, 0, 1, 0, 0, -1, 0],
#         [0, 0, 0, 0, 0, 1, 0, 0, 0, -1],
#         [0, 0, 0, 0, 0, 0, 1, -1, 0, 0],
#         [0, 0, 0, 0, 0, 0, 1, 0, -1, 0],
#         [0, 0, 0, 0, 0, 0, 1, 0, 0, -1],
#         [0, 0, 0, 0, 0, 0, 0, 1, -1, 0],
#         [0, 0, 0, 0, 0, 0, 0, 1, 0, -1],
#         [0, 0, 0, 0, 0, 0, 0, 0, 1, -1],
#      ],
# )

def get_default_device():
  if torch.cuda.is_available():
    print("Run on Cuda")
    return torch.device("cuda")
  else:
    return torch.device("cpu")

def to_device(data, device):
  if isinstance(data, (list, tuple)):
    return [to_device(x, device) for x in data]
  return data.to(device, non_blocking= True)


def generate_binary_batches(binary_class, dataset):
  batches = dict()
  for ind, sub_targ in enumerate(binary_class):
    print("{} the binary class is: {}".format(ind, sub_targ))
    sub_targ = list(sub_targ)

    new_data = []
    for indx, (img, targets) in enumerate(dataset):
      for i in range(len(sub_targ)):
        if targets == sub_targ[i]:
          new_data.append((img, sub_targ[i]))
    batches["Batch_{}".format(ind)] = new_data

    return batches


def get_binary_classes(n_classes):
  custom_class = []
  count = 0
  for i in range(len(n_classes)):
    for j in range(len(n_classes)):
      if n_classes[i] == n_classes[i] + n_classes[j]:
        pass
      else:
        if  n_classes[i] + n_classes[j] < 10:
          #custom_class.append((count, (n_classes[i], n_classes[i] + n_classes[j])))
          custom_class.append((n_classes[i], n_classes[i] + n_classes[j]))
          count +=1
        else:
          pass
  custom_classes = tuple(custom_class)
  return custom_classes


def generate_models( model, n_channels, n_classes):
  models = []
  n_classifiers = n_classes*(n_classes - 1) / 2
  for i in range(round(n_classifiers)):
    model =model
    models.append((i, model))

  models = tuple(models)
  return models


def some_infos(loader):
  dataiter = iter(loader)
  images, labels = dataiter.next()
  print("Type of DataLoader: {} \nShape of Images: {} \nShape of Labels: {}".format(
      type(images),
      images.shape,
      labels.shape,
      ))
  for idx, (data, target) in enumerate(loader):
    real_label = torch.max(target, 1)[1].data.squeeze()
    print("Batch Labels:  " , real_label)
    break


def test_CIFAR_10(test_batch, workers):

    transform_test = transforms.ToTensor()
    mean = [0.4914, 0.4822, 0.4465]
    std=[0.2023, 0.1994, 0.2010]

    testset = torchvision.datasets.CIFAR10(root='./datasets/data/cifar10', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch, pin_memory=True,
                                             shuffle=False, num_workers=workers)
    num_classes = 10
    return testloader, num_classes, mean, std

def test_SVHN(test_batch, workers):

    transform_test = transforms.ToTensor()
    mean = [0.4376821, 0.4437697, 0.47280442]
    std = [0.19803012, 0.20101562, 0.19703614]

    testset = torchvision.datasets.SVHN(root='./datasets/data/SVHN', split='test',
                                        download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch, pin_memory=True,
                                             shuffle=False, num_workers=workers)
    num_classes =10
    return testloader, num_classes, mean, std

def test_GTSRB(test_batch, workers):

    transform_test = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.RandomCrop(96, padding=4),
        transforms.RandomHorizontalFlip(1),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        ])
    mean=[0.3337, 0.3064, 0.3171]
    std=[0.2672, 0.2564, 0.2629]

    testset = torchvision.datasets.GTSRB(root='./datasets/data/GTSRB', split='test',
                                        download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch, pin_memory=True,
                                             shuffle=False, num_workers=workers)
    num_classes =10
    return testloader, num_classes, mean, std

def test_STL_10(test_batch, workers):
    transform_test = transforms.Compose([
        transforms.Resize((96, 96)),
        # transforms.RandomCrop(96, padding=4),
        # transforms.RandomHorizontalFlip(1),
        # transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])

    mean = [0.4272, 0.4200, 0.3885]
    std = [0.2395, 0.2363, 0.2357]


    testset = torchvision.datasets.STL10(root='./datasets/data/STL_10', split='test',
                                        download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch, pin_memory=True,
                                             shuffle=False, num_workers=workers)
    num_classes =10
    return testloader, num_classes, mean, std

def load_test_dataset_un_normalized(datasetName, test_batch, workers):
    if datasetName =='CIFAR_10':
        testloader, num_classes, mean, std = test_CIFAR_10(test_batch, workers)
    elif datasetName == "SVHN":
        testloader, num_classes, mean, std = test_SVHN(test_batch, workers)
    elif datasetName == 'GTSRB':
        testloader, num_classes, mean, std = test_GTSRB(test_batch, workers)
    elif datasetName == 'STL_10':
        testloader, num_classes, mean, std = test_STL_10(test_batch, workers)
    else:
        raise NameError("Dataset " +datasetName + "is not supported by this application!")
    return testloader, num_classes, mean, std

def get_mean_std(loader=None):
    mean, std = 0.0, 0.0
    for img, lbl in loader:

        batch_samples = img.size(0)  # batch size (the last batch can have smaller size!)
        img = img.view(batch_samples, img.size(1), -1)
        mean += img.mean(2).sum(0)
        std += img.std(2).sum(0)
    mean /= len(loader.dataset)
    std /= len(loader.dataset)
    return mean,std

def label_to_ovo(y,n_classes, matrix):
    target_trans = (lambda y: torch.sum((torch.Tensor(torch.zeros(n_classes, dtype=torch.float).scatter_(0, torch.tensor(y),value=1)) * matrix).T, axis=0))

    y_target = y.clone().detach()

    y_true = [target_trans(y_target[i]) for i in range(len(y))]
    return torch.tensor(np.array([torch.Tensor.cpu(y_true[i]).detach().numpy() for i in range(len(y_true))]))




class Transforms_OvO(object):
    def __init__(self, args):
        self.n_classes = args.n_classes

    def grab_ovo_classes(self):
        return int(self.n_classes * (self.n_classes - 1) / 2)

    def generate_trans_matrix(self):
        n_classif = round(self.n_classes * (self.n_classes - 1) / 2)
        c = 0
        N = np.zeros(shape=(self.n_classes, n_classif))
        for i in range(N.shape[0]):
            k = c + (self.n_classes - (i + 1))
            N[i, c: k] = 1
            np.fill_diagonal(N.T[c:k, (i + 1):self.n_classes], -1)
            c = c + (self.n_classes - (i + 1))

        N = N.astype('int32')
        return N.T

    def label_to_ovo(self, y, n_classes):
        matrix = self.generate_trans_matrix()
        target_trans = (lambda y: torch.sum((torch.Tensor(torch.zeros(n_classes, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)) * matrix).T,axis=0))

        y_target = y.clone().detach()

        y_true = [target_trans(y_target[i]) for i in range(len(y))]
        return torch.tensor(np.array([torch.Tensor.cpu(y_true[i]).detach().numpy() for i in range(len(y_true))]))

    def reverse_OvO_labels(self, y):
        transform = self.generate_trans_matrix()
        reverse_trans = transform.T
        return [torch.sum(torch.from_numpy(reverse_trans).cpu() * y[i].cpu(), axis=1) for i in range(len(y))]

    def get_preds(self, y_scores=None):
        y_scores = self.reverse_OvO_labels(y=y_scores)
        y_scores = torch.tensor(
            np.array([torch.Tensor.cpu(y_scores[i]).detach().numpy() for i in range(len(y_scores))]))
        _, preds = torch.max(y_scores, dim=1)
        return preds

    def accuracy(self, y_true, y_scores):
        labels, preds = 0, 0
        y_true = self.reverse_OvO_labels(y=y_true)
        y_scores = self.reverse_OvO_labels(y=y_scores)
        y_true = torch.tensor(np.array([torch.Tensor.cpu(y_true[i]).detach().numpy() for i in range(len(y_true))]))
        y_scores = torch.tensor(
            np.array([torch.Tensor.cpu(y_scores[i]).detach().numpy() for i in range(len(y_scores))]))
        _, labels = torch.max(y_true, dim=1)

        _, preds = torch.max(y_scores, dim=1)

        return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count