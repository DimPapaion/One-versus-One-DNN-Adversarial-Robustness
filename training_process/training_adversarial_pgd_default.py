"""
Created on Wed Jan 23 10:15:27 2019

@author: aamir-mustafa
This is Part 1 file for replicating the results for Paper:
    "Adversarial Defense by Restricting the Hidden Space of Deep Neural Networks"
Here a ResNet model is trained with Softmax Loss for 164 epochs.
"""

# Essential Imports
import os
import sys
import argparse
import datetime
import time
import os.path as osp
import torch.backends.cudnn as cudnn
import numpy as np
from Config import *
from models import resnet_ova, resnet_96x96  # Imports the ResNet Model
from models.MC_SVDD import *
from datasets import *
from Utils import *

parser = argparse.ArgumentParser("Softmax Training", parents=[get_args_parser()])
parser.add_argument('--dataset', type=str, default='STL_10')
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--train-batch', default=64, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=64, type=int, metavar='N',
                    help='test batchsize')
# parser.add_argument('--lr', type=float, default=0.1, help="learning rate for model")
# parser.add_argument('--schedule', type=int, nargs='+', default=[142, 230, 360],
#                     help='Decrease learning rate at these epochs.')
# parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
# parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
#                     help='momentum')
# parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
#                     metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--max-epoch', type=int, default=400)
parser.add_argument('--eval-freq', type=int, default=10)
parser.add_argument('--print-freq', type=int, default=200)
# parser.add_argument('--gpu', type=str, default='1')  # gpu to be used
# parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--save-dir', type=str, default='log')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

__, ___, mean, std = load_test_dataset_un_normalized(args.dataset, args.test_batch, args.workers)

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

def attack(model, criterion, img, label, eps, attack_type, iters):
    adv = un_normalize(img.detach())
    adv.requires_grad = True

    if attack_type == 'fgsm':
        iterations = 1
    else:
        iterations = iters

    if attack_type == 'pgd':
        step = 2 / 255
    else:
        step = eps / iterations

        noise = 0

    for j in range(iterations):
        _, _, _, out_adv = model(normalize(adv.clone()))
        loss = criterion(out_adv, label)
        loss.backward()

        if attack_type == 'mim':
            adv_mean = torch.mean(torch.abs(adv.grad), dim=1, keepdim=True)
            adv_mean = torch.mean(torch.abs(adv_mean), dim=2, keepdim=True)
            adv_mean = torch.mean(torch.abs(adv_mean), dim=3, keepdim=True)
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

    return adv.detach()



def main():
    args.seed = 1
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()

    if args.use_cpu: use_gpu = False



    if use_gpu:
        print("Currently using GPU: {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

    # Data Loading
    CD = CustomDatasets(args, transform_matrix=None)
    dataset = CD._load_Dataset()
    loaders = CD.make_loaders()
    trainloader, testloader = loaders['Train_load'], loaders['Test_load']
    num_classes = args.n_classes
    print('==> Preparing dataset ' +args.dataset)

    # Loading the Model

    model = resnet(num_classes=args.n_classes, depth=110)
    print(model)

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    for p in model.named_parameters():
        print(p[0])

    criterion1 = nn.CrossEntropyLoss()
    # Need to check Adam here
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)


    filename = './models/checkpoints/final/STL_10_softmax_model.pth.tar'
    checkpoint = torch.load(filename)

    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict = checkpoint['optimizer_model']
    start_time = time.time()

    for epoch in range(args.max_epoch):
        adjust_learning_rate(optimizer, epoch)

        print("==> Epoch {}/{}".format(epoch + 1, args.max_epoch))
        print('LR: %f' % (state['lr']))

        train(trainloader, model, criterion1, optimizer, epoch, use_gpu, num_classes)

        if args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0 or (epoch + 1) == args.max_epoch:
            print("==> Test")  # Tests after every 10 epochs
            acc, err = test(model, testloader, use_gpu, num_classes, epoch)
            print("Accuracy (%): {}\t Error rate (%): {}".format(acc, err))


            checkpoint = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
                          'optimizer_model': optimizer.state_dict(),
                          }
            torch.save(checkpoint, './models/checkpoints/' +args.dataset +'PGD_Adversarial_Training_final.pth.tar')

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))


def train(trainloader, model, criterion1, optimizer, epoch, use_gpu, num_classes):
    model.train()

    losses_xent = AverageMeter()

    # Batch-wise Training
    for batch_idx, (data, labels) in enumerate(trainloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()



        model.eval()
        eps = np.random.uniform(0.02, 0.05)
        adv = attack(model, criterion1, data, labels, eps=eps, attack_type='pgd',
                     iters=10)  # Generates Batch-wise Adv Images
        adv.requires_grad = False

        adv = normalize(adv)
        adv = adv.cuda()
        true_labels_adv = labels
        data = adv
        data = torch.cat((data, adv), 0)
        labels = torch.cat((labels, true_labels_adv))
        model.train()

        feats_128, feats_256, feats_1024, outputsLinear  = model(data)
        loss_xent = criterion1(outputsLinear, labels)  # cross-entropy loss calculation

        optimizer.zero_grad()

        loss_xent.backward()
        optimizer.step()

        losses_xent.update(loss_xent.item(), labels.size(0))  # AverageMeter() has this param

        if (batch_idx + 1) % args.print_freq == 0:
            print("Batch {}/{}\t Loss {:.6f} ({:.6f}) " \
                  .format(batch_idx + 1, len(trainloader), losses_xent.val, losses_xent.avg))


def test(model, testloader, use_gpu, num_classes, epoch):
    model.eval()

    correct, total = 0.0, 0.0

    with torch.no_grad():
        for data, labels in testloader:
            if use_gpu:
                data, labels = data.cuda(), labels.cuda()
            feats_128, feats_256, feats_1024, outputsSVM = model(data)

            # Preds linear
            predictionsSVM = outputsSVM.data.max(1)[1]
            total += labels.size(0)
            correct += (predictionsSVM == labels.data).sum()


    acc = correct * 100. / total
    err = 100. - acc


    return acc, err


def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


if __name__ == '__main__':
    main()



























