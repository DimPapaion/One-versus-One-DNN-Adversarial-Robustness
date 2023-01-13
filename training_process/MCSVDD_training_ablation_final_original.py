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
from Config import *
from models import resnet_ovo, resnet_96x96_ovo  # Imports the ResNet Model
from models.MC_SVDD_ablation import *
from datasets import *
from Utils import *

parser = argparse.ArgumentParser("MCSVDD_Training", parents=[get_args_parser()])
parser.add_argument('--dataset', type=str, default='CIFAR_100')

parser.add_argument('--train-batch', default=32, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=32, type=int, metavar='N',
                    help='test batchsize')

parser.add_argument('--lr_svdd', type=float, default=0.05, help="learning rate for model")

parser.add_argument('--max-epoch', type=int, default=400)
parser.add_argument('--eval-freq', type=int, default=10)
parser.add_argument('--print-freq', type=int, default=200)


parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--ablation', type=str, default='n', help="Choices: [std, m, m+p, m+n, p+n, p, n]")

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}


# %%

def main():
    args.seed=1
    args.gpu = str(2)
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
    # trainloader, testloader, num_classes = load_dataset(args.dataset, args.train_batch, args.test_batch, args.workers)
    print('==> Preparing dataset ' +args.dataset)
    CD = CustomDatasets(args, transform_matrix=None)
    dataset = CD._load_Dataset()
    loaders = CD.make_loaders()
    trainloader, testloader = loaders['Train_load'], loaders['Test_load']
    num_classes = 10
    # Loading the Model

    model = resnet_ovo.resnet(num_classes=num_classes, depth=110)
    SVDDmodel_256 = MulticlassSVDDClassifier(256, num_classes)
    SVDDmodel_1024 = MulticlassSVDDClassifier(1024, num_classes)
    if use_gpu:
        model = nn.DataParallel(model).cuda()
        SVDDmodel_256 = nn.DataParallel(SVDDmodel_256)
        SVDDmodel_1024 = nn.DataParallel(SVDDmodel_1024)

    for p in model.named_parameters():
        print(p[0])
    for p in SVDDmodel_256.named_parameters():
        print(p[0])
    for p in SVDDmodel_1024.named_parameters():
        print(p[0])



    criterion1 = nn.CrossEntropyLoss()
    criterion2 = AdvancedSVDDLoss(num_classes,args.ablation)
    # Need to check Adam here
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    SVDD_256_optimizer = torch.optim.SGD(SVDDmodel_256.parameters(), lr=args.lr_svdd, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    SVDD_1024_optimizer = torch.optim.SGD(SVDDmodel_1024.parameters(), lr=args.lr_svdd, momentum=args.momentum,
                                weight_decay=args.weight_decay)

    filename = './STL_10_softmax_model.pth.tar'
    checkpoint = torch.load(filename)

    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict = checkpoint['optimizer_model']

    start_time = time.time()

    for epoch in range(args.max_epoch):
        adjust_learning_rate(optimizer, epoch)
        # The grey lines did not run on the saved model. Maybe try again sometime
       # adjust_learning_rate(SVDD_256_optimizer, epoch)
       # adjust_learning_rate(SVDD_1024_optimizer, epoch)

        print("==> Epoch {}/{}".format(epoch + 1, args.max_epoch))
        print('LR: %f' % (state['lr']))

        train(trainloader, model, SVDDmodel_256, SVDDmodel_1024, criterion1, criterion2, optimizer, SVDD_256_optimizer, SVDD_1024_optimizer, epoch, use_gpu, num_classes)

        if args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0 or (epoch + 1) == args.max_epoch:
            print("==> Test")  # Tests after every 10 epochs
            acc, err, accSvdd1, errSvdd1, accSvdd2, errSvdd2 = test(model, SVDDmodel_256, SVDDmodel_1024, testloader, use_gpu, num_classes, epoch)
            print("Accuracy (%): {}\t Error rate (%): {}".format(acc, err))
            print("Accuracy SVDD1(%): {}\t Error rate (%): {}".format(accSvdd1, errSvdd1))
            print("Accuracy SVDD2(%): {}\t Error rate (%): {}".format(accSvdd2, errSvdd2))

            checkpoint = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
                          'optimizer_model': optimizer.state_dict(),
                          'svdd1_state_dict' : SVDDmodel_256.state_dict(),
                          'svdd2_state_dict': SVDDmodel_1024.state_dict(),
                          'svdd1_optimizer_state_dict': SVDD_256_optimizer.state_dict(),
                          'svdd2_optimizer_state_dict': SVDD_1024_optimizer.state_dict()
                          }
            torch.save(checkpoint, './' +args.dataset +'_MC_SVDD_pretrained_final' +args.ablation+'.tar')

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))


def train(trainloader, model, svdd1, svdd2, criterion1, criterion2, optimizer, svdd1optimizer, svdd2optimizer, epoch, use_gpu, num_classes):
    model.train()
    svdd1.train()
    svdd2.train()
    losses_xent = AverageMeter()
    losses_svdd1 = AverageMeter()
    losses_svdd2 = AverageMeter()
    losses_overall = AverageMeter()

    # Batch-wise Training
    for batch_idx, (data, labels) in enumerate(trainloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        feats_128, feats_256, feats_1024, outputsLinear  = model(data)
        loss_xent = criterion1(outputsLinear, labels)  # cross-entropy loss calculation

        outputsSVDD1 = svdd1(feats_256)
        outputsSVDD2 = svdd2(feats_1024)

        # SVDD1 loss
        loss_svdd1 = criterion2(outputsSVDD1[0],outputsSVDD1[1],outputsSVDD1[2], labels)

        # SVDD2 loss
        loss_svdd2 = criterion2(outputsSVDD2[0],outputsSVDD2[1],outputsSVDD2[2], labels)
        #if epoch>10:
        loss = loss_xent + loss_svdd1 + loss_svdd2
        #else:
        #loss =  loss_xent + loss_svdd1 + margin_loss

        optimizer.zero_grad()
        svdd1optimizer.zero_grad()
        svdd2optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        svdd1optimizer.step()
        svdd2optimizer.step()

        losses_xent.update(loss_xent.item(), labels.size(0))  # AverageMeter() has this param
        losses_svdd1.update(loss_svdd1.item(), labels.size(0))
        losses_svdd2.update(loss_svdd2.item(), labels.size(0))
        losses_overall.update(loss.item(), labels.size(0))

        if (batch_idx + 1) % args.print_freq == 0:
            print("Batch {}/{}\t Loss {:.6f} ({:.6f}) " \
                  .format(batch_idx + 1, len(trainloader), losses_xent.val, losses_xent.avg))
            print("Batch {}/{}\t Loss {:.6f} ({:.6f}) " \
                  .format(batch_idx + 1, len(trainloader), losses_svdd1.val, losses_svdd1.avg))
            print("Batch {}/{}\t Loss {:.6f} ({:.6f}) " \
                  .format(batch_idx + 1, len(trainloader), losses_svdd2.val, losses_svdd2.avg))
            print("Batch {}/{}\t Loss {:.6f} ({:.6f}) " \
                  .format(batch_idx + 1, len(trainloader), losses_overall.val, losses_overall.avg))


def test(model, svdd1, svdd2, testloader, use_gpu, num_classes, epoch):
    model.eval()
    svdd1.eval()
    svdd2.eval()

    correct, total, correctSVDD1, correctSVDD2 = 0, 0, 0, 0

    with torch.no_grad():
        for data, labels in testloader:
            if use_gpu:
                data, labels = data.cuda(), labels.cuda()
            feats_128, feats_256, feats_1024, outputsSVM = model(data)
            predsSVDD1 = svdd1(feats_256)
            predsSVDD2 = svdd2(feats_1024)
            # Preds linear
            predictionsSVM = outputsSVM.data.max(1)[1]
            total += labels.size(0)
            correct += (predictionsSVM == labels.data).sum()

            # Preds SVDD1
            predictionsSVDD1 = predsSVDD1[0].data.max(1)[1]
            correctSVDD1 += (predictionsSVDD1 == labels.data).sum()

            # Preds SVDD2
            predictionsSVDD2 = predsSVDD2[0].data.max(1)[1]
            correctSVDD2 += (predictionsSVDD2 == labels.data).sum()

    acc = correct * 100. / total
    err = 100. - acc

    accSVDD1 = correctSVDD1 * 100. /total
    errSVDD1 = 100. - accSVDD1

    accSVDD2 = correctSVDD2 * 100. / total
    errSVDD2 = 100. - accSVDD2

    return acc, err, accSVDD1, errSVDD1, accSVDD2, errSVDD2


def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


if __name__ == '__main__':
    main()



























