from Utils import *
import os
import sys
import argparse
import datetime
import time
import os.path as osp
import tqdm as tqdm
import torch.backends.cudnn as cudnn
from Dataset import *
from models import resnet_ova,resnet_96x96  # Imports the ResNet Model
from models.MC_SVDD_ablation import *
# from datasets import *
from fit import *
#from utils.utils import AverageMeter, Logger
from Config import *


parser = argparse.ArgumentParser("MCSVDD_Training", parents=[get_args_parser()])
parser.add_argument('--dataset', type=str, default='STL-10')
parser.add_argument('-j', '--workers', default=1, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
# parser.add_argument('--lr', type=float, default=0.1, help="learning rate for model")
parser.add_argument('--lr_svdd_1', type=float, default=0.05, help="learning rate for model")
parser.add_argument('--lr_svdd_2', type=float, default=0.001, help="learning rate for model")
# parser.add_argument('--schedule', type=int, nargs='+', default=[80, 220, 320],
#                     help='Decrease learning rate at these epochs.')
# parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
# parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
#                     help='momentum')
parser.add_argument('--weight-decay_SVDD', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--max-epoch', type=int, default=400)
parser.add_argument('--eval-freq', type=int, default=10)
parser.add_argument('--print-freq', type=int, default=200)
# parser.add_argument('--gpu', type=str, default='3')  # gpu to be used
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--ablation', type=str, default='std', help="Choices: [std, m, m+p, m+n, p+n, p, n]")

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}


# %%



def main():
    torch.manual_seed(args.seed)
    args.gpu = str(2)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    #sys.stdout = Logger(osp.join(args.save_dir, 'final/log_' + args.dataset +'_MC_SVDD_' +args.ablation +'.txt'))

    if use_gpu:
        print("Currently using GPU: {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

        # Initialize One versus One Setup.


        # Create the dataset:

    CD = CustomDatasets(args, transform_matrix=None)
    dataset = CD._load_Dataset()
    loaders = CD.make_loaders()

    trainloader, testloader = loaders['Train_load'], loaders['Test_load']
    print("{} dataset: \n{}".format(args.dataset_name, dataset))


    model = resnet_96x96.resnet(num_classes=args.n_classes, depth=110)
    # model = MyResnet101_(n_classes=10)
    # model = resnet_full.resnet101(num_classes = args.n_classes)
    print(model)
    SVDDmodel_256 = MulticlassSVDDClassifier(256, args.n_classes)
    SVDDmodel_1024 = MulticlassSVDDClassifier(1024, args.n_classes)
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

    checkpoint = torch.load("./models/checkpoints/final/'STL_10_softmax_model.pth.tar")
    model.load_state_dict(checkpoint['state_dict'])
    model = model.module

    # model = model.cuda()

    criterion1 = nn.CrossEntropyLoss()
    criterion2 = AdvancedSVDDLoss(num_classes = args.n_classes, ablation=args.ablation)
    # Need to check Adam here
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    SVDD_256_optimizer = torch.optim.SGD(SVDDmodel_256.parameters(), lr=args.lr_svdd_1, momentum=args.momentum,
                                weight_decay=args.weight_decay_SVDD)
    SVDD_1024_optimizer = torch.optim.SGD(SVDDmodel_1024.parameters(), lr=args.lr_svdd_1, momentum=args.momentum,
                                weight_decay=args.weight_decay_SVDD)



    start_time = time.time()

    best_acc = 0.0

    for epoch in range(args.max_epoch):
        adjust_learning_rate(optimizer, epoch)
        # adjust_learning_rate_svdd(optimizer=SVDD_1024_optimizer, epoch= epoch)
        # adjust_learning_rate_svdd(optimizer=SVDD_256_optimizer, epoch=epoch)

        print("==> Epoch {}/{}".format(epoch + 1, args.max_epoch))
        print('LR: %f' % (state['lr']))
        print('LR svdd: %f' % (state['lr_svdd_1']))

        train(trainloader, model, SVDDmodel_256, SVDDmodel_1024, criterion1, criterion2, optimizer, SVDD_256_optimizer, SVDD_1024_optimizer, epoch, use_gpu, args.n_classes)
        acc, err, accSvdd1, errSvdd1, accSvdd2, errSvdd2 = test(model, SVDDmodel_256, SVDDmodel_1024, testloader,use_gpu, args.n_classes, epoch)
        if args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0 or (epoch + 1) == args.max_epoch:
            print("==> Test")  # Tests after every 10 epochs

            print("Accuracy (%): {}\t Error rate (%): {}".format(acc, err))
            print("Accuracy SVDD1(%): {}\t Error rate (%): {}".format(accSvdd1, errSvdd1))
            print("Accuracy SVDD2(%): {}\t Error rate (%): {}".format(accSvdd2, errSvdd2))

            checkpoint = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
                          'optimizer_model': optimizer.state_dict(),
                          'svdd1_state_dict': SVDDmodel_256.state_dict(),
                          'svdd2_state_dict': SVDDmodel_1024.state_dict(),
                          'svdd1_optimizer_state_dict': SVDD_256_optimizer.state_dict(),
                          'svdd2_optimizer_state_dict': SVDD_1024_optimizer.state_dict()
                          }
            torch.save(checkpoint, './' + args.dataset + '_MC_SVDD_OVA_v8' + args.ablation + '.tar')
            print("chceckpoint is saved.")

        if best_acc < acc:
            best_acc = acc

            print("Accuracy (%): {}\t Error rate (%): {}".format(acc, err))
            print("Accuracy SVDD1(%): {}\t Error rate (%): {}".format(accSvdd1, errSvdd1))
            print("Accuracy SVDD2(%): {}\t Error rate (%): {}".format(accSvdd2, errSvdd2))


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
    for data, labels in tqdm(trainloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()



        feat_128, feats_256, feats_1024, outputsLinear  = model(data)
        # print("Features 128 {} with shape {}".format(feats_128, feats_128.shape))
        # print("Features 256 {} with shape {}".format(feats_256, feats_256.shape))
        # print("Features 1024 {} with shape {}".format(feats_1024, feats_1024.shape))
        # print("Features output {} with shape {}".format(outputsLinear, outputsLinear.shape))




        outputsSVDD1 = svdd1(feats_256)
        outputsSVDD2 = svdd2(feats_1024)
        # print("Output SVDD1 {},\n {},\n {} with shape {}, \n {}, \n{}".format(outputsSVDD1[0], outputsSVDD1[1], outputsSVDD1[2],
        #                                                                       outputsSVDD1[0].shape,
        #                                                                       outputsSVDD1[1].shape,
        #                                                                       outputsSVDD1[2].shape))
        # print("Features output {} with shape {}".format(outputsLinear, outputsLinear.shape))

        loss_xent = criterion1(outputsLinear, labels) # OvO Losss
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
        # batch_idx +=1
        # if (batch_idx + 1) % args.print_freq == 0:
        #     print("Batch {}/{}\t Loss {:.6f} ({:.6f}) " \
        #           .format(batch_idx + 1, len(trainloader), losses_xent.val, losses_xent.avg))
        #     print("Batch {}/{}\t Loss {:.6f} ({:.6f}) " \
        #           .format(batch_idx + 1, len(trainloader), losses_svdd1.val, losses_svdd1.avg))
        #     print("Batch {}/{}\t Loss {:.6f} ({:.6f}) " \
        #           .format(batch_idx + 1, len(trainloader), losses_svdd2.val, losses_svdd2.avg))
        #     print("Batch {}/{}\t Loss {:.6f} ({:.6f}) " \
        #           .format(batch_idx + 1, len(trainloader), losses_overall.val, losses_overall.avg))
    print(
        "Train Loss Model: {:.10f} / Avg Loss: ({:.10f}), \n Train Loss SVDD1: {:.10f} / Avg Loss: ({:.10f}), \n Train Loss SVDD2: {:.10f} / Avg Loss: ({:.10f}), \n Overall Train Loss: {:.10f} / Average Loss {:.10f}".format(
            losses_xent.val,losses_xent.avg,
            losses_svdd1.val, losses_svdd1.avg,
            losses_svdd2.val, losses_svdd2.avg,
            losses_overall.val, losses_overall.avg))


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
            # predictionsSVM = get_preds(outputsSVM)
            _, predictions = torch.max(outputsSVM, dim=1)
            # # predictionsSVM = predictionsSVM.data.max(1)[1]
            total += labels.size(0)
            correct += (predictions == labels.data).sum()

            # # Preds SVDD1
            # predictionsSVDD1 = get_preds(predsSVDD1[0])
            predictionsSVDD1 = predsSVDD1[0].data.max(1)[1]
            correctSVDD1 += (predictionsSVDD1 == labels.data).sum()
            # acc1 += accuracy(torch.Tensor.cpu(labels).detach(), torch.Tensor.cpu(predsSVDD1[0]).detach())
            # # Preds SVDD1
            # # Preds SVDD2

            # predictionsSVDD2 = get_preds(predsSVDD2[0])
            predictionsSVDD2 = predsSVDD2[0].data.max(1)[1]
            correctSVDD2 += (predictionsSVDD2 == labels.data).sum()
            # acc2 += accuracy(torch.Tensor.cpu(labels).detach(), torch.Tensor.cpu(predsSVDD2[0]).detach())
            # # Preds SVDD1

    accur = correct * 100. / total
    err = 100. - accur

    # accSVDD1 = acc1 * 100. /float(len(testloader))
    # errSVDD1 = 100. - acc1
    accSVDD1 = correctSVDD1 * 100. /total
    errSVDD1 = 100. - accSVDD1
    # accSVDD2 = acc2 * 100. / float(len(testloader))
    # errSVDD2 = 100. - acc2
    accSVDD2 = correctSVDD2 * 100. / total
    errSVDD2 = 100. - accSVDD2


    return accur, err, accSVDD1, errSVDD1, accSVDD2, errSVDD2


def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

def adjust_learning_rate_svdd(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr_svdd_1'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr_svdd_1'] = state['lr_svdd_1']


if __name__ == '__main__':
    main()
