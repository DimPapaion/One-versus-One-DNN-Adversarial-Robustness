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

from models.resnet import *  # Imports the ResNet Model
from models.MC_SVDD import *
from datasets import *
from utils.utils import AverageMeter, Logger

parser = argparse.ArgumentParser("Softmax Training")
parser.add_argument('--dataset', type=str, default='CIFAR_100')
parser.add_argument('-j', '--workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', type=float, default=0.1, help="learning rate for model")
parser.add_argument('--schedule', type=int, nargs='+', default=[142, 230, 360],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--max-epoch', type=int, default=400)
parser.add_argument('--eval-freq', type=int, default=10)
parser.add_argument('--print-freq', type=int, default=200)
parser.add_argument('--gpu', type=str, default='1')  # gpu to be used
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--save-dir', type=str, default='log')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}


# %%

def main():
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    sys.stdout = Logger(osp.join(args.save_dir, 'log_' + args.dataset +'_Softmax Training' + '.txt'))

    if use_gpu:
        print("Currently using GPU: {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

    # Data Loading
    trainloader, testloader, num_classes = load_dataset(args.dataset, args.train_batch, args.test_batch, args.workers)
    print('==> Preparing dataset ' +args.dataset)

    # Loading the Model

    model = resnet(num_classes=num_classes, depth=110)

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    for p in model.named_parameters():
        print(p[0])

    criterion1 = nn.CrossEntropyLoss()
    # Need to check Adam here
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)

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
            torch.save(checkpoint, './models/checkpoints/' +args.dataset +'_softmax_model.pth.tar')

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



























