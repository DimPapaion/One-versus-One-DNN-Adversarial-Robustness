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
from models import resnet_ovo, resnet_96x96, resnet_ova, OvO_Loss_CE
from models.MC_SVDD_ablation import *
# from datasets import *

#from utils.utils import AverageMeter, Logger
from Config import *


parser = argparse.ArgumentParser("Inference", parents=[get_args_parser()])
parser.add_argument('--dataset', type=str, default='STL_10')
parser.add_argument('-j', '--workers', default=1, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=100, type=int, metavar='N',
                    help='test batchsize')




args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}


# %%

mean = [0.4272, 0.4200, 0.3885]
std = [0.2395, 0.2363, 0.2357]
# mean = [0.5, 0.5, 0.5]
# std = [0.5, 0.5, 0.5]
def normalize(t):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0]) / std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1]) / std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2]) / std[2]
    return t
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
    args.seed = 1
    torch.manual_seed(args.seed)
    args.gpu = str(2)
    args.gpu = '0'
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


        # Create the dataset:

    # CD = CustomDatasets(args, transform_matrix=None)
    # dataset = CD._load_Dataset()
    # loaders = CD.make_loaders()
    #
    # trainloader, testloader = loaders['Train_load'], loaders['Test_load']
    # print("{} dataset: \n{}".format(args.dataset_name, dataset))
    testloader, num_classes, mean, std = load_test_dataset_un_normalized(args.dataset, args.test_batch, args.workers)

    model = resnet_96x96.resnet(num_classes=args.n_classes, depth=110)
    # model = OvO_Loss_CE.MyResnet101_(n_classes=10)
    # model = resnet_full.resnet101(num_classes = args.n_classes)
    # model = torchvision.models.resnet101(num_classes= args.n_classes)
    print(model)
    #
    #)
    # model = model.cuda()





    if use_gpu:
        model = nn.DataParallel(model).cuda()

    checkpoint = torch.load("./STL_10_MC_SVDD_OVA_v2std.tar")
    model.load_state_dict(checkpoint['state_dict'])
    # if use_gpu:
    #     model = nn.DataParallel(model).cuda()







    start_time = time.time()

    print('<<<<-------Starting Inference process -------->>>')
    acc, err = test(model, testloader,use_gpu, args.n_classes)


    print("Accuracy (%): {}\t Error rate (%): {}".format(acc, err))


    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

def test(model, testloader, use_gpu, num_classes):
    model.eval()


    correct, total, correctSVDD1, correctSVDD2 = 0, 0, 0, 0

    with torch.no_grad():
        for data, labels in testloader:
            if use_gpu:
                data, labels = data.cuda(), labels.cuda()
            feats_128, feats_256, feats_1024,outputsSVM = model(normalize(data))

            # Preds linear
            # predictionsSVM = get_preds(outputsSVM)
            _, predictions = torch.max(outputsSVM, dim=1)
            # # predictionsSVM = predictionsSVM.data.max(1)[1]
            total += labels.size(0)
            correct += (predictions == labels.data).sum()


    accur = correct * 100. / total
    err = 100. - accur




    return accur, err

if __name__ == '__main__':
    main()

