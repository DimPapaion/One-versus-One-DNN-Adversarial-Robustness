"""
Created on Sun Mar 24 17:51:08 2019

@author: aamir-mustafa
"""
import matplotlib.pyplot as plt
import torch
import pytorch_ssim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from models.resnet_ova import *
from datasets import *
import models
from torch.autograd import Variable
import os.path as osp
import torch.backends.cudnn as cudnn
import argparse
import json
import time
import datetime
# To be deleted
# from models.resnet import *  # Imports the ResNet Model
import numpy as np
from Dataset import *
# from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity
from models import resnet_96x96, resnet_ova, resnet_ovo, resnet_96x96_ovo, OvO_Loss_CE


parser = argparse.ArgumentParser("Robustness evaluation", parents=[get_args_parser()])
parser.add_argument('--dataset', type=str, default='CIFAR_10')
parser.add_argument('--trained_model', type=str, default='PCL')
parser.add_argument('-j', '--workers', default=1, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--test-batch', default=64, type=int, metavar='N',
                    help='test batchsize')

parser.add_argument('--attack_type', type=str, default='fgsm')
"""
Adversarial Attack Options: fgsm, bim, mim, pgd
"""

args = parser.parse_args()
args.seed = 1
state = {k: v for k, v in args._get_kwargs()}

torch.manual_seed(args.seed)
cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_gpu = torch.cuda.is_available()

# Loading Test Data (Un-normalized)
testloader, n_classes, mean, std = load_test_dataset_un_normalized(args.dataset, args.test_batch, args.workers)

# mean=[0.4914, 0.4822, 0.4465]
# std=[0.2023, 0.1994, 0.2010]

# mean=[0.4376821, 0.4437697, 0.47280442]
# std=[0.19803012, 0.20101562, 0.19703614]

num_classes_ovo = int(n_classes * (n_classes-1) / 2)
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
def get_preds(y_scores=None):
    y_scores = reverse_OvO_labels(y=y_scores, transform=M)
    y_scores = torch.tensor(np.array([torch.Tensor.cpu(y_scores[i]).detach().numpy() for i in range(len(y_scores))]))
    _, preds = torch.max(y_scores, dim=1)
    return preds
# Attacking Images batch-wise
def reverse_OvO_labels( y, transform):
    reverse_trans = transform.T
    return [torch.sum(torch.from_numpy(reverse_trans).cuda() * y[i].cuda(), axis=1) for i in range(len(y))]
#model = resnet(num_classes=num_classes,depth=110)
# to get proposed
if args.trained_model == "OvO_MCSVDD_FS" or args.trained_model == "OvO_MCSVDD_FT" or args.trained_model == "OvO":
    if args.dataset == "GTSRB" or args.dataset == 'STL_10':
        model = resnet_96x96_ovo.resnet(num_classes=num_classes_ovo, depth=110)
        model = nn.DataParallel(model).cuda()
        model = model.cuda()
    else:
        model = resnet_ovo.resnet(num_classes=num_classes_ovo, depth=110)
        model = nn.DataParallel(model).cuda()

else:
    if args.dataset == "GTSRB" or args.dataset == 'STL_10':
        model = resnet_96x96.resnet(num_classes=args.n_classes, depth=110)
        model = nn.DataParallel(model).cuda()
    else:
        model = resnet_ova.resnet(num_classes=args.n_classes, depth=110)
        model = nn.DataParallel(model).cuda()

# Model directory
my_model = ""
if args.trained_model=="MCSVDD":
    my_model =my_model.join(['models/checkpoints/', args.dataset, '_MCSVDD_model.pth.tar'])
if args.trained_model=="MCSVDD_std":
    my_model =my_model.join(['models/checkpoints/final/', args.dataset, '_MC_SVDD_std.tar'])
if args.trained_model=="orthonormal":
    my_model =my_model.join(['models/checkpoints/final/', args.dataset, '_MC_SVDD_orthonornal_std.tar'])
if args.trained_model=="orthonormal_pre":
    my_model =my_model.join(['models/checkpoints/final/', args.dataset, '_MC_SVDD_pretrained_orthonornal_std.tar'])
elif args.trained_model=="MCSVDD_pretrained":
    my_model =my_model.join(['models/checkpoints/final/', args.dataset, '_MC_SVDD_pretrainedstd.tar'])
elif args.trained_model=="advtrain":
    my_model =my_model.join(['models/checkpoints/final/', args.dataset, 'advtrainPGD_MC_SVDD_pretrained_v2std.tar'])
elif args.trained_model == "advtrainDef":
    my_model = my_model.join(['models/checkpoints/', args.dataset, 'PGD_Adversarial_Training.pth.tar'])
elif args.trained_model=="softmax":
    my_model =my_model.join(['models/checkpoints/final/', args.dataset, '_softmax_model.pth.tar'])
elif args.trained_model=="PCL":
    my_model =my_model.join(['models/checkpoints/', args.dataset, '_PCL_pretrained.pth.tar'])
elif args.trained_model=="CenterLoss":
    my_model =my_model.join(['models/checkpoints/final/', args.dataset, '_CenterLoss_model.pth.tar'])
elif args.trained_model=="OvA":
    my_model =my_model.join(['./Random_trainined_models/models_ova/STL_10_MC_SVDD_pretrainedstd.tar'])
elif args.trained_model=="OvO_MCSVDD_FS":
    my_model =my_model.join(['./models_ovo/checkpoints/final/',args.dataset, '_MC_SVDD_OVOstd.tar'])
elif args.trained_model == "OvO_MCSVDD_FT":
    my_model = my_model.join(['./models_ovo/checkpoints/final/',args.dataset, '_MC_SVDD_OVO_pretrained_OVA_softmax_std.tar'])


checkpoint = torch.load(my_model)
model.load_state_dict(checkpoint['state_dict'])
model.eval()
# if args.dataset =='SVHN' and args.trained_model == 'OvO':
#     model = MyResnet101_(n_classes=45).cuda()
#     checkpoint = torch.load('./Exp_1SVHN-ResNet101.pt')
#     model.load_state_dict(checkpoint)
#     model.eval()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Attacking Images batch-wise
def attack(model, criterion, img, label, eps, attack_type, iters, i):

    adv = img.detach()
    adv.requires_grad = True
    mse = 0
    ssim_ = 0
    if attack_type == 'fgsm':
        iterations = 1
    else:
        iterations = iters

    if attack_type == 'pgd':
        step = 4 / 255
    else:
        step = eps / iterations

        noise = 0

    for j in range(iterations):
        _,_,_,out_adv = model(normalize(adv.clone()))

        if args.trained_model == 'OvO' or args.trained_model == 'OvO_MCSVDD_FS' or args.trained_model == 'OvO_MCSVDD_FT':
            pred_adv = get_preds(out_adv)
            label_ = get_preds(label)
            if torch.sum(pred_adv == label_.cpu()).item() == 0:
                break
        else:
            _, pred_adv = torch.max(out_adv, dim=1)


            if torch.sum(pred_adv == label).item()==0:
                break
        loss = criterion(out_adv, label)
        loss.backward()

        if attack_type == 'mim':
            adv_mean= torch.mean(torch.abs(adv.grad), dim=1,  keepdim=True)
            adv_mean= torch.mean(torch.abs(adv_mean), dim=2,  keepdim=True)
            adv_mean= torch.mean(torch.abs(adv_mean), dim=3,  keepdim=True)
            adv.grad = adv.grad / adv_mean
            noise = noise + adv.grad
        else:
            noise = adv.grad

        # Optimization step
        adv.data = adv.data + step * noise.sign()
#        adv.data = adv.data + step * adv.grad.sign()

        if attack_type == 'pgd':
            adv.data = torch.where(adv.data > img.data + eps, img.data + eps, adv.data)
            adv.data = torch.where(adv.data < img.data - eps, img.data - eps, adv.data)
        adv.data.clamp_(0.0, 1.0)

        adv.grad.data.zero_()
        mse = ((img.detach() - adv.detach())**2).mean()



        img1 = Variable(img, requires_grad=False)
        img2 = Variable(adv, requires_grad=True)

        ssim_ = pytorch_ssim.ssim(img1=img1.detach(), img2=img2.detach(), window_size=11).mean()

    return adv.detach(), mse,  ssim_,j

# Loss Criteria
if args.trained_model == 'OvO' or args.trained_model == 'OvO_MCSVDD_FS' or args.trained_model == 'OvO_MCSVDD_FT' :
    criterion = OvO_Loss_CE.OvOLoss(num_classes=45) #
else:
    criterion = nn.CrossEntropyLoss() #

def calculate_metrics(args, testloader, criterion, eps):
    adv_acc = 0
    clean_acc = 0
    total =0
    mse=0
    ssim=0
    print("<<=======Starting the process for attack type: {} with noise: {}======>>".format(args.attack_type, eps))
    # eps = 8#8/255 # Epsilon for Adversarial Attack
    # model = model.to(device)
    for i, (img, label) in enumerate(testloader):
        img, label = img.to(device), label.to(device)

        label_ovo = label_to_ovo(y=torch.Tensor.cpu(label), n_classes=10, matrix=M).to(device)
        total+=label.size(0)

        _,_,_,out = model(normalize(img.clone().detach()))

        if args.trained_model == 'OvO' or args.trained_model == 'OvO_MCSVDD_FS' or args.trained_model == 'OvO_MCSVDD_FT':
            predictions = get_preds(out)
            clean_acc += torch.sum(predictions == label.cpu()).item()
        else:
            _, predictions = torch.max(out, dim=1)
            clean_acc += torch.sum(predictions == label).item()

        adv, mse_batch,ssim_batch , iterations= attack(model, criterion, img, label, eps=eps, attack_type= args.attack_type, iters= 100, i=i)
        _,_,_,out_adv = model(normalize(adv.clone().detach()))


        if args.trained_model == 'OvO' or args.trained_model == 'OvO_MCSVDD_FS' or args.trained_model == 'OvO_MCSVDD_FT':
            predictions_adv = get_preds(out_adv)
            # _, predictions_adv = torch.max(out_adv, dim=1)
            adv_acc += torch.sum(predictions_adv == label.cpu()).item()
        else:
            _, predictions_adv = torch.max(out_adv, dim=1)
            adv_acc += torch.sum(predictions_adv == label).item()

        mse +=mse_batch
        ssim += ssim_batch
        # print(iterations)
        # print('Batch: {0}'.format(i))

    print('Clean accuracy:{0:.3%}\t Adversarial accuracy:{1:.3%}'.format(clean_acc / total, adv_acc / total))
    print('MSE:{0:.5f}\t'.format(mse / (i+1)))
    print('SSIM:{0:.5f}\t'.format(ssim / (i+1)))
    results = dict()
    results["attack_type"] = args.attack_type
    results["test_accuracy"] = clean_acc/total
    results["robustness"] = adv_acc/total
    results["mse"] = (mse/(i+1)).item()
    results["ssim"] = (ssim/(i+1)).item()
    results["dataset"] = args.dataset
    results["trained_model"] = args.trained_model
    return results


eps_list = [0.5, 5]
attack_list = ['fgsm', 'pgd', 'bim', 'mim']

start_time = time.time()
for i in range(len(eps_list)):
    eps = eps_list[i]

    for k in range(len(attack_list)):
        args.attack_type = attack_list[k]
        results = calculate_metrics(args, testloader=testloader, criterion=criterion, eps = eps)

elapsed = round(time.time() - start_time)
elapsed = str(datetime.timedelta(seconds=elapsed))
print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))









