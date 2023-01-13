

from Models import *
import os
import sys
import argparse
import datetime
import time
import os.path as osp
import tqdm as tqdm
import torch.backends.cudnn as cudnn
from Dataset import *
from models.resnet import *  # Imports the ResNet Model
from models.MC_SVDD_ablation import *
# from datasets import *
from fit import *
#from utils.utils import AverageMeter, Logger
from Config import *
import resnet_full

parser = argparse.ArgumentParser("MCSVDD_Training", parents=[get_args_parser()])
parser.add_argument('--dataset', type=str, default='GTSRB')
parser.add_argument('-j', '--workers', default=1, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')
# parser.add_argument('--lr', type=float, default=0.1, help="learning rate for model")
parser.add_argument('--lr_svdd', type=float, default=0.05, help="learning rate for model")
# parser.add_argument('--schedule', type=int, nargs='+', default=[80, 220, 320],
#                     help='Decrease learning rate at these epochs.')
# parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
# parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
#                     help='momentum')
# parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
#                     metavar='W', help='weight decay (default: 1e-4)')
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

def main():
    torch.manual_seed(args.seed)

    args.gpu = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False
    args.dataset_name = 'STL-10'
    args.n_classes = 10
    #sys.stdout = Logger(osp.join(args.save_dir, 'final/log_' + args.dataset +'_MC_SVDD_' +args.ablation +'.txt'))

    if use_gpu:
        print("Currently using GPU: {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

        # Initialize One versus One Setup.
    transf = Transforms_OvO(args)
    num_classes_ovo = transf.grab_ovo_classes()
    transform_matrix = transf.generate_trans_matrix()

        # Create the dataset:

    CD = CustomDatasets(args, transform_matrix=transform_matrix)
    dataset = CD._load_Dataset()
    loaders = CD.make_loaders()

    trainloader, testloader = loaders['Train_load'], loaders['Test_load']
    print("{} dataset: \n{}".format(args.dataset_name, dataset))


    model = resnet(num_classes=10, depth=110)
    # model = MyResnet101_(n_classes=10)
    # model = resnet_full.resnet101(num_classes = args.n_classes)
    # model = torchvision.models.resnet101(num_classes= args.n_classes)
    print(model)
    model = model.cuda()
    #
    if use_gpu:
        model = nn.DataParallel(model).cuda()

    checkpoint = torch.load("./STL-10_PCL_AdvTrain_PGD.pth.tar")
    model.load_state_dict(checkpoint['state_dict'])

    # model = model.cuda()






    start_time = time.time()

    print('<<<<-------Starting Inference process -------->>>')
    acc, err = test(model, testloader,use_gpu, transf,args.n_classes)


    print("Accuracy (%): {}\t Error rate (%): {}".format(acc, err))


    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

def test(model, testloader, use_gpu,transf ,  num_classes):
    model.eval()


    correct, total, correctSVDD1, correctSVDD2 = 0, 0, 0, 0

    with torch.no_grad():
        for data, labels in testloader:
            if use_gpu:
                data, labels = data.cuda(), labels.cuda()
            feats_128, feats_256, feats_1024, outputsSVM = model(data)

            # Preds linear
            # labels_n = transf.get_preds(labels)
            # predictions = transf.get_preds(outputsSVM)
            _, predictions = torch.max(outputsSVM, dim=1)
            # # predictionsSVM = predictionsSVM.data.max(1)[1]
            total += labels.size(0)
            correct += (predictions == labels).sum()


    accur = correct * 100. / total
    err = 100. - accur




    return accur, err

if __name__ == '__main__':
    main()


