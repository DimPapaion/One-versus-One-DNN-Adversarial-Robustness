"""
Created on Sun Mar 24 17:51:08 2019

@author: aamir-mustafa
"""
from Utils import *
from Config import *
from models import resnet_ova, resnet_ovo, resnet_96x96, resnet_96x96_ovo
from models import OvO_Loss_CE
from Dataset import *
import torch.backends.cudnn as cudnn
import argparse
import datetime
import time
from tqdm import tqdm

from Models import *


parser = argparse.ArgumentParser("Robustness evaluation", parents=[get_args_parser()])
parser.add_argument('--dataset', type=str, default='STL_10')
parser.add_argument('--trained_model', type=str, default='OvO')
parser.add_argument('--robust_model', type=str, default='OvO')
parser.add_argument('-j', '--workers', default=1, type=int,
                    help="number of data loading workers (default: 4)")
parser.add_argument('--test-batch', default=16, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--attack_type', type=str, default='fgsm')
"""
Adversarial Attack Options: fgsm, bim, mim, pgd.
"""

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
args.seed = 1
torch.manual_seed(args.seed)
cudnn.benchmark = True
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
# use_gpu = torch.cuda.is_available()

# Loading Test Data (Un-normalized)
testloader, num_classes, mean, std = load_test_dataset_un_normalized(args.dataset, args.test_batch, args.workers)

args.n_classes = 10


transforms_ovo = Transforms_OvO(args)
num_classes_ovo = transforms_ovo.grab_ovo_classes()


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


def attack(args,model, criterion, img, label, eps, attack_type, iters):
    adv = img.detach()
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
        if args.dataset == 'SVHN' and args.trained_model == "OvO":
            out_adv = model(normalize(adv.clone()))
        else:
            _,_,_,out_adv = model(normalize(adv.clone()))
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

def load_model(trained_model):
    if trained_model == "OvO_MCSVDD_FS" or trained_model == "OvO_MCSVDD_FT" or trained_model == "OvO":
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
        #
    # model = resnet_96x96.resnet(num_classes=args.n_classes, depth=110)
    # model = nn.DataParallel(model).cuda()

    # model = model.cuda()
    my_model = ""
    if trained_model=="MCSVDD":
        my_model =my_model.join(['models/checkpoints/', args.dataset, '_MCSVDD_model.pth.tar'])
    if trained_model=="MCSVDD_std":
        my_model =my_model.join(['models/checkpoints/final/Done/', args.dataset, '_MC_SVDD_std.tar'])
    if trained_model=="orthonormal":
        my_model =my_model.join(['models/checkpoints/final/', args.dataset, '_MC_SVDD_orthonornal_std.tar'])
    if trained_model=="orthonormal_pre":
        my_model =my_model.join(['models/checkpoints/final/', args.dataset, '_MC_SVDD_pretrained_orthonornal_std.tar'])
    if trained_model=="MCSVDD_pretrained":
        # my_model = my_model.join(['./STL-10_OVA_V6.tar'])
        my_model =my_model.join(['./models/checkpoints/final/', args.dataset, '_MC_SVDD_pretrainedstd.tar'])
    if trained_model == "MCSVDD_ICIP":
        my_model = my_model.join(['models/checkpoints/final/', args.dataset, '_MC_SVDD_pretrained_ICIP_std.tar'])
    if trained_model == "MCSVDD_ICIP_v2":
        my_model = my_model.join(['models/checkpoints/final/', args.dataset, '_MC_SVDD_pretrained_ICIP_v2std.tar'])
    if trained_model == "MCSVDD_ICIP_v3":
        my_model = my_model.join(['models/checkpoints/final/', args.dataset, '_MC_SVDD_pretrained_ICIP_v3std.tar'])
    if trained_model=="advtrain":
        my_model =my_model.join(['models/checkpoints/final/', args.dataset, 'advtrainPGD_MC_SVDD_pretrained_v2std.tar'])
    if trained_model == "advtrainDef":
        my_model = my_model.join(['models/checkpoints/final/', args.dataset, '_PGD_Adversarial_Training.pth.tar'])
    if trained_model == "softmax":

        my_model = my_model.join(['./models/checkpoints/final/',args.dataset, '_softmax_model.pth.tar'])
    if trained_model=="PCL":
        # my_model = my_model.join(['STL_10_PCL_pretrained.pth.tar'])
        my_model =my_model.join(['models/checkpoints/final/', args.dataset, '_PCL_pretrained.pth.tar'])
    if trained_model=="CenterLoss":
        my_model =my_model.join(['models/checkpoints/final/', args.dataset, '_CenterLoss_model.pth.tar'])
    if trained_model == "OvO":
        my_model = my_model.join(['./models_ovo/checkpoints/final/', args.dataset,'_OVO.tar'])
        # my_model = my_model.join(['./STL-10_MC_SVDD_OVO_pretrained_v9std.tar'])
    if trained_model == "OvO_MCSVDD_FS":
        my_model = my_model.join(['./models_ovo/checkpoints/final/', args.dataset,'_MC_SVDD_OVOstd.tar'])
    if trained_model=="OvO_MCSVDD_FT":
        my_model =my_model.join(['./models_ovo/checkpoints/final/', args.dataset,'_MC_SVDD_OVO_pretrained_OVA_softmax_std.tar'])
        # my_model = my_model.join(['./STL-10_MC_SVDD_OVO_pretrained_v9std.tar'])


    checkpoint = torch.load(my_model)
    if args.dataset == 'STL_10':
        if  trained_model=='OvO_MCSVDD_FT':
            model.module.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


# Model directory
# attack_model = resnet(num_classes=num_classes, depth=110)
# attack_model.cuda()
# my_model=''
#
# my_model =my_model.join(['models/checkpoints/final/', args.dataset, '_softmax_model.pth.tar'])
# checkpoint = torch.load(my_model, map_location='gpu')
#
# attack_model.load_state_dict(checkpoint['state_dict'])
# attack_model.eval()
#
# print(attack_model)
# attack_model = load_model(args.trained_model)
# robust_model = load_model(args.robust_model)

# attack_model = MyResnet101_(n_classes=45)
# checkpoint = torch.load('./Exp_1SVHN-ResNet101.pt')
# attack_model.load_state_dict(checkpoint)
#

def construct_models(args):
    if args.dataset =="SVHN":
        if args.trained_model == "OvO":

            attack_model = MyResnet101_(n_classes=45)
            checkpoint = torch.load('./models_ovo/checkpoints/final/Exp_1SVHN-ResNet101.pt')
            attack_model.load_state_dict(checkpoint)

            robust_model = load_model(args.robust_model)
        elif args.robust_model == "OvO":

            robust_model = MyResnet101_(n_classes=45)
            checkpoint = torch.load('./models_ovo/checkpoints/final/Exp_1SVHN-ResNet101.pt')
            robust_model.load_state_dict(checkpoint)

            attack_model = load_model(args.trained_model)
        else:
            attack_model = load_model(args.trained_model)
            robust_model = load_model(args.robust_model)

    else:
        attack_model = load_model(args.trained_model)
        robust_model = load_model(args.robust_model)

    robust_model.eval()
    attack_model.eval()
#
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    robust_model.to(device)
    attack_model.to(device)
    return robust_model, attack_model





# Loss Criteria
if args.trained_model == 'OvO' or args.trained_model == 'OvO_MCSVDD_FS' or args.trained_model == 'OvO_MCSVDD_FT':
    criterion = OvO_Loss_CE.OvOLoss(num_classes=num_classes_ovo) #
else:
    criterion = nn.CrossEntropyLoss() #
# eps = 0.05#0.03#8 / 255  # Epsilon for Adversarial Attack
def calculate_adv_attacks(args, testloader, criterion, eps,robust_model, attack_model, transforms = transforms_ovo):
    adv_acc = 0
    clean_acc = 0
    total = 0
    # eps = 0.05#0.03#8 / 255  # Epsilon for Adversarial Attack
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for img, label in tqdm(testloader):
        img, label = img.to(device), label.to(device)

        # if trained_model=="OvO":
        label_ovo = transforms.label_to_ovo(y=torch.Tensor.cpu(label), n_classes=args.n_classes).to(device)

        # print(normalize(img.clone().detach())[3].argmax(dim=-1))
        with torch.no_grad():
            if args.dataset == 'SVHN' and args.robust_model=='OvO':
                out = robust_model(normalize(img.clone().detach()))
            else:
                _,_,_,out = robust_model(normalize(img.clone().detach()))

        if args.robust_model == "OvO_MCSVDD_FS" or args.robust_model == "OvO_MCSVDD_FT" or args.robust_model == "OvO":
            preds = transforms.get_preds(y_scores=out)
            clean_acc += torch.sum(preds == label.cpu()).item()
        # print(preds)
        else:
        # print("Print output:", torch.Tensor.cpu(out))
        # print("label: ", label)
            _, preds = torch.max(out, dim=1)
            clean_acc += torch.sum(preds == label).item()


        total += label.size(0)
        if args.trained_model == "OvO_MCSVDD_FS" or args.trained_model == "OvO_MCSVDD_FT" or args.trained_model == "OvO":
            adv = attack(args,attack_model, criterion, img, label_ovo, eps=eps, attack_type=args.attack_type, iters=10)
        else:
            adv = attack(args,attack_model, criterion, img, label, eps=eps, attack_type=args.attack_type, iters=10)

        with torch.no_grad():
            if args.dataset == 'SVHN' and args.robust_model=='OvO':
                out_adv = robust_model(normalize(adv.clone().detach()))
            else:
                _,_,_,out_adv = robust_model(normalize(adv.clone().detach()))

        if args.robust_model == "OvO_MCSVDD_FS" or args.robust_model == "OvO_MCSVDD_FT" or args.robust_model == "OvO":
            preds_adv = transforms.get_preds(y_scores=out_adv)
            adv_acc += torch.sum(preds_adv == label.cpu()).item()
        else:
            _, preds_adv = torch.max(out_adv, dim=1)
            adv_acc += torch.sum(preds_adv == label).item()
        # print('Batch: {0}'.format(i))

        torch.cuda.empty_cache()
    print('Noise: {}, Attack_type {}, Clean accuracy:{}%, Adversarial accuracy:{}%'.format(eps, args.attack_type, (clean_acc / total) * 100, (adv_acc / total) * 100))
    return

eps_list = [0.01, 0.03, 0.05]
attack_list = ['fgsm', 'pgd', 'bim', 'mim']
trained_=['softmax', 'advtrainDef', 'PCL', 'MCSVDD_pretrained', 'OvO', 'OvO_MCSVDD_FS',  'OvO_MCSVDD_FT']
robust_  =['softmax', 'advtrainDef', 'PCL', 'MCSVDD_pretrained', 'OvO', 'OvO_MCSVDD_FS',  'OvO_MCSVDD_FT']
start_time = time.time()

for j in range(len(trained_)):
    # args.trained_model =trained_[j]
    args.trained_model = 'OvO_MCSVDD_FS'
    for l in range(len(robust_)):
        # args.robust_model = robust_[l]
        args.robust_model = 'OvO_MCSVDD_FS'
        if args.trained_model == args.robust_model:
            print('Black box attack is performed with attack model: {} to robust model: {} .....'.format(args.trained_model, args.robust_model))
            robust_model, attack_model = construct_models(args)
            for i in range(len(eps_list)):
                eps = eps_list[i]

                for k in range(len(attack_list)):
                    args.attack_type = attack_list[k]
                    calculate_adv_attacks(args, testloader=testloader, criterion=criterion,robust_model=robust_model, attack_model=attack_model, eps = eps)

elapsed = round(time.time() - start_time)
elapsed = str(datetime.timedelta(seconds=elapsed))
print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))







