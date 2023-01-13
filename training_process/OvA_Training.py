from Config import *
from Dataset import *
import os
from models import resnet_ova, resnet_96x96
from Utils import *
from fit import *
from Models import *

def main(args):
    state = {k: v for k, v in args._get_kwargs()}
    # Initialized Cuda:

    torch.manual_seed(1)
    args.gpu = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda:' + args.gpu if torch.cuda.is_available() else 'cpu')

    args.dataset_name = 'STL-10'
    args.n_classes = 10
    args.lr = 0.01
    # Initialize One versus One Setup.


    # Create the dataset:

    CD = CustomDatasets(args, transform_matrix=None)
    dataset = CD._load_Dataset()
    loaders = CD.make_loaders()

    print("{} dataset: \n{}".format(args.dataset_name, dataset))
    print("device: ", device)

    #Visualize dataset
    show_dataset(args, dataset=dataset['Train'])

    # Clarify training hyperparameters


    #Load Pretrained model.
    model_ovo = resnet(num_classes=args.n_classes, depth=110)
    # model_ovo = MyResnet101_(n_classes=10)
    # model_ovo = resnet_full.resnet101(num_classes = args.n_classes)
    # model_ovo = torchvision.models.resnet101(num_classes = args.n_classes)
    model_ovo = nn.DataParallel(model_ovo).cuda()

    # checkpoint = torch.load("./models/checkpoints/final/CIFAR_100_softmax_model.pth.tar")
    # model_ovo.load_state_dict(checkpoint['state_dict'])
    # model_ovo = model_ovo.module
    # model_ovo.fcf = nn.Linear(in_features=1024, out_features=num_classes_ovo)



    optimizer_ovo = torch.optim.SGD(params=model_ovo.parameters(), lr=args.lr, momentum=args.momentum, weight_decay= args.weight_decay)
    criterion_ovo = nn.CrossEntropyLoss()

    if args.is_lr_step_adjusted:
        scheduler = None
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_ovo, T_max=200)


    training_obg = BaseClassifierOvA(args, model = model_ovo, optimizer = optimizer_ovo, criterion = criterion_ovo,
                                  scheduler = scheduler, train_dl = loaders['Train_load'], valid_dl = loaders['Valid_load'],
                                  test_dl = loaders['Test_load'], device = device, transform_matrix = None, state=  state,
                                  model2=None)
    if args.is_trainable:

        [tr_loss, tr_acc], [val_loss, val_acc] = training_obg.train_model()

        test_loss, test_acc, y_pred, y_true = training_obg.test_model()
    else:
        test_loss, test_acc, y_pred, y_true = training_obg.test_model()





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Cifar-10 Test Set.", parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)