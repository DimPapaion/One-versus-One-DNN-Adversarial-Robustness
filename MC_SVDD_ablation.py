"""
Created on Sep 7 2020

@author: Mygdalis Vasileios
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# One-class SVDD classifier
class OvAClassifier(nn.Module):
    def __init__(self, feat_dim = 1568, num_classes=10):
        super(OvAClassifier, self).__init__()
        self.fc = nn.Linear(in_features=feat_dim, out_features=num_classes)
    def forward(self, x):
        out = self.fc(x)
        return out


class SvddClassifier(nn.Module):
    def __init__(self, feat_dim=1568):
        super(SvddClassifier, self).__init__()
#        self.R = nn.Parameter(1.0*torch.ones(1), requires_grad=True)
#        self.R = nn.Parameter(1.1*torch.ones(1), requires_grad=True)
        self.R = nn.Parameter(1*torch.ones(1), requires_grad=True)
        self.a = nn.Parameter(torch.rand(feat_dim), requires_grad=True)

    def forward(self, x):
        distances = torch.cdist(x,torch.unsqueeze(self.a,0),2)
        output = torch.pow(self.R,2) - distances
        positive_outputs = distances/ (torch.pow(self.R,2)+0.001)
        #positive_outputs = distances
        negative_outputs = torch.pow(self.R, 2) / (distances+0.001)
        #negative_outputs = torch.pow(self.R,2) / (distances)


        return torch.squeeze(output), torch.squeeze(positive_outputs), torch.squeeze(negative_outputs)


class MulticlassSVDDClassifier(nn.Module):
    def __init__(self, feat_dim, noOfClasses):
        super(MulticlassSVDDClassifier, self).__init__()
        self.noOfClasses = noOfClasses

        self.svdd = nn.ModuleList()

        for class_iter in range(noOfClasses):
            self.svdd.append(SvddClassifier(feat_dim))

    def forward(self, x):
        outputs1 = []
        outputs2 = []
        outputs3 = []

        for class_iter in range(self.noOfClasses):
            s1, s2, s3 = self.svdd[class_iter].forward(x)
            outputs1.append(s1)
            outputs2.append(s2)
            outputs3.append(s3)
        output1 = torch.stack(outputs1, 1)
        output2 = torch.stack(outputs2, 1)
        output3 = torch.stack(outputs3, 1)
        return output1, output2, output3


class MarginMultiClassSVDDLoss(nn.Module):
    def __init__(self, num_classes=2):
        super(MarginMultiClassSVDDLoss, self).__init__()
        self.numClasses = num_classes

    def forward(self, x, y, size_average=True):
        one_hot_labels = torch.zeros(len(y), self.numClasses).scatter_(1, y.cpu().unsqueeze(1), 1.).cuda()
        targets = one_hot_labels.clone()
        targets[one_hot_labels == 0] = -1.0
        losses = F.relu(-targets * x)
        return losses.mean() if size_average else losses.sum()


class AdvancedSVDDLoss(nn.Module):
    def __init__(self, num_classes=2, ablation='std'):
        super(AdvancedSVDDLoss, self).__init__()
        self.numClasses = num_classes
        self.ablation = ablation

    def forward(self, x, positiveOutputs, negativeOutputs, y, size_average=True):
        one_hot_labels = torch.zeros(len(y), self.numClasses).scatter_(1, y.cpu().unsqueeze(1), 1.).cuda()
        targets = one_hot_labels.clone()
        # targets[one_hot_labels == 0] = -1.0
        targets[one_hot_labels == 0] = -1.0
        margin_loss = F.relu(-targets * x)
        positive_targets = 1/2*(targets+1).clone()
        negative_targets = -1/2*(targets-1).clone()
        positive_loss = positive_targets * positiveOutputs
        negative_loss = negative_targets * negativeOutputs

        if self.ablation=='std':
            losses = margin_loss + positive_loss + negative_loss
        elif self.ablation=='m':
            losses = margin_loss
        elif self.ablation == 'm+p':
            losses = margin_loss + positive_loss
        elif self.ablation == 'm+n':
            losses = margin_loss + negative_loss
        elif self.ablation == 'p':
            losses = positive_loss
        elif self.ablation == 'n':
            losses = negative_loss
        elif self.ablation == 'p+n':
            losses = positive_loss + negative_loss


        return losses.mean() if size_average else losses.sum()
