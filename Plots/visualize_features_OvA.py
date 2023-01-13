import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from datasets import *
import os
import os.path as osp
import torch.backends.cudnn as cudnn
import argparse
import torch.nn.functional as F
import json
# To be deleted
import models.resnet_96x96 as res # Imports the ResNet Model
from Dataset import *
from Config import *


parser = argparse.ArgumentParser("Robustness evaluation", parents=[get_args_parser()])
parser.add_argument('--dataset', type=str, default='STL_10')
parser.add_argument('--trained_model', type=str, default='MCSVDD_pretrained_')
parser.add_argument('-j', '--workers', default=64, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--test-batch', default=16, type=int, metavar='N',
                    help='test batchsize')

parser.add_argument('--attack_type', type=str, default='bim')
parser.add_argument('--plot', type=str, default=True)
"""
Adversarial Attack Options: fgsm, bim, mim, pgd
"""

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

torch.manual_seed(args.seed)
cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_gpu = torch.cuda.is_available()

# Loading Test Data (Un-normalized)
# testloader, num_classes, mean, std = load_test_dataset_for_visualization_un_normalized(args.dataset, args.test_batch, args.workers)
testloader, num_classes, mean, std = load_test_dataset_un_normalized(args.dataset, args.test_batch, args.workers)
#
# CD = CustomDatasets(args, transform_matrix = None)
# dataset = CD._load_Dataset()
# print("{} dataset: {} \n".format(args.dataset_name, dataset))
#
# loaders = CD.make_loaders()
# print('==> Preparing dataset ' +args.dataset)
# trainloader, testloader = loaders['Train_load'], loaders['Test_load']
# num_classes = 10

# # mean=[0.4914, 0.4822, 0.4465]
# # std=[0.2023, 0.1994, 0.2010]
# mean=[0.4272, 0.4200, 0.3885]
# std=[0.2395, 0.2363, 0.2357]


def normalize(t):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0]) / std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1]) / std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2]) / std[2]

    return t


def un_normalize(t):
    t[:, 0, :, :] = (t[:, 0, :, :] * std[0]) + mean[0]
    t[:, 1, :, :] = (t[:, 1, :, :] * std[1]) + mean[1]
    t[:, 2, :, :] = (t[:, 2, :, :] * std[2]) + mean[2]

    return t

def test(model, testloader, use_gpu, num_classes):
    model.eval()
    correct, total = 0, 0
    all_features, all_labels = [], []

    with torch.no_grad():
        for data, labels in testloader:

            if use_gpu:
                data, labels = data.cuda(), labels.cuda()
            feats_128, feats_256, feats_1024, outputs = model(normalize(data))
            predictions = outputs.data.max(1)[1]
            total += labels.size(0)
            correct += (predictions == labels.data).sum()

            if use_gpu:
                all_features.append(feats_1024[predictions == labels.data,:].data.cpu().numpy())
                all_labels.append(labels[predictions == labels.data].data.cpu().numpy())
            else:
                all_features.append(feats_1024[predictions == labels.data,:].data.numpy())
                all_labels.append(labels[predictions == labels.data].data.numpy())

    if args.plot:
        all_features = np.concatenate(all_features, 0)
        all_labels = np.concatenate(all_labels, 0)

        all_features = PCA_numpy(all_features)
        #all_features = all_features[:,1:3]
        plot_features(all_features, all_labels, num_classes)

    acc = correct * 100. / total
    err = 100. - acc
    return acc, err


def PCA_numpy(data, n_components=2):
    # 1nd step is to find covarience matrix
    data_vector = []
    for i in range(data.shape[1]):
        data_vector.append(data[:, i])

    cov_matrix = np.cov(data_vector)

    # 2rd step is to compute eigen vectors and eigne values
    eig_values, eig_vectors = np.linalg.eig(cov_matrix)
    eig_values = np.reshape(eig_values, (len(cov_matrix), 1))

    # Make pairs
    eig_pairs = []
    for i in range(len(eig_values)):
        eig_pairs.append([np.abs(eig_values[i]), eig_vectors[:, i]])

    eig_pairs.sort(reverse=True, key=(lambda x: x[0]))
    #eig_pairs.reverse()

    # This PCA is only for 2 components
    reduced_data = np.hstack(
        (eig_pairs[0][1].reshape(len(eig_pairs[0][1]), 1), eig_pairs[1][1].reshape(len(eig_pairs[0][1]), 1)))

    return data.dot(reduced_data)


def plot_features(features, labels, num_classes):
    """Plot features on 2D plane.
    Args:
        features: (num_instances, num_features).
        labels: (num_instances).
    """
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    for label_idx in range(num_classes):
        plt.scatter(
            features[labels==label_idx, 0],
            features[labels==label_idx, 1],
            c=colors[label_idx],
            s=1,
        )
    #plt.gca().update(dict(title='Space', xlim=(-5,5), ylim =(-5,5)))

    # plt.legend(['airplanes', 'cars', 'birds', 'cats', 'deer', 'dogs', 'frogs', 'horses', 'ships', 'trucks'], loc='best')


    # dirname = os.getcwd()
    save_name = osp.join('STL_10/', args.dataset + args.trained_model + '.png')
    plt.savefig(save_name, bbox_inches='tight')
    plt.close()

#model = resnet(num_classes=num_classes,depth=110)
# to get proposed
model = res.resnet(num_classes=num_classes,depth=110)
if True:
    model = nn.DataParallel(model).cuda()

# Model directory
my_model = ""
if args.trained_model=="MCSVDD":
    my_model =my_model.join(['models/checkpoints/', args.dataset, '_MC_SVDD_std.tar'])
elif args.trained_model=="MCSVDD_pretrained":
    my_model =my_model.join(['models/checkpoints/final/', args.dataset, '_MC_SVDD_pretrainedstd.tar'])
elif args.trained_model=="softmax":
    my_model =my_model.join(['models/checkpoints/final/', args.dataset, '_softmax_model.pth.tar'])
elif args.trained_model=="PCL":
    my_model =my_model.join(['models/checkpoints/', args.dataset, '_PCL_pretrained.pth.tar'])
elif args.trained_model=="CenterLoss":
    my_model =my_model.join(['models/checkpoints/final/', args.dataset, '_CenterLoss_model.pth.tar'])
elif args.trained_model == "MCSVDD_ICIP_v2":
        my_model = my_model.join(['models/checkpoints/final/', args.dataset, '_MC_SVDD_pretrained_ICIP_v2std.tar'])
elif args.trained_model == "MCSVDD_pretrained_":
        my_model = my_model.join(['Random_trainined_models/models_ova/', args.dataset_name, '_MC_SVDD_OVA_pretrained_v2std.tar'])
checkpoint = torch.load(my_model)
model.load_state_dict(checkpoint['state_dict'])
model.eval()
print(model)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

acc, err = test(model,testloader,True,num_classes)
print(acc)
