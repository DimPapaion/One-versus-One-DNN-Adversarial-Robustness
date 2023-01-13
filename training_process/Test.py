from Utils import *

from Dataset import *









def config(type_con=None, model=None,model2=None, Params=None, loaders=None, dataset=None):
    """
    Type of name type_con requires arguments of the form
     - "loaders" if you want to create the dataset loaders.
     - "train" if you want to create the parameters for the training phase.
    """
    if type_con == "loaders":
        DatasetParams = dict()
        DatasetParams['path'] = "./"
        DatasetParams['target_type'] = 'OVO'
        DatasetParams['dataset_name'] = "CIFAR10"
        DatasetParams['batch_size'] =128
        DatasetParams['n_classes'] = 10
        DatasetParams['transform'] = True
        DatasetParams['Validation'] = True
        DatasetParams['Random_Seed'] = 10
        DatasetParams['Valid_Size'] = 0.2
        DatasetParams['Shuffle'] = True
        DatasetParams['Pin_Memory'] = True
        DatasetParams['Matrix'] = M
        return DatasetParams

    elif type_con == "models":
        # vgg16 = MyVgg16(dataset_name='CIFAR10')
        res101 = MyResnet101_(n_classes=45)

        # mobile = MyMobileV2(dataset_name='CIFAR10')
        # shuffle = MyShuffleNetv2(dataset_name="CIFAR10",trained=True)
        ParamsMod = dict()
        # ParamsMod["Cifar10-ResNet20"] = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=False)
        ParamsMod["Cifar10-ResNet101"] = res101 #.construct(n_classes=45, dataset_name='CIFAR10')

        return ParamsMod

    elif type_con == "train":
        Parameters = dict()
        Parameters['model'] = Params[model]
        Parameters['model2'] =None #model2
        Parameters['model_name'] = model
        Parameters['path'] = "./" + str('Exp5_') + str(Parameters['model_name']) + ".pt"
        Parameters['epochs'] = 400
        Parameters['device'] = get_default_device()
        Parameters['criterion'] = OvOLoss(num_classes=45)

        Parameters['optimizer'] =torch.optim.SGD(Parameters['model'].parameters(), lr=0.01,momentum=0.9,weight_decay=5e-4) # torch.optim.Adam(params=Parameters['model'].parameters(), lr=0.001,weight_decay=5e-4)
        Parameters['scheduler'] = torch.optim.lr_scheduler.StepLR(Parameters['optimizer'], step_size=200)#torch.optim.lr_scheduler.CosineAnnealingLR(Parameters['optimizer'], T_max=400)
        Parameters['train_dl'] = loaders['Train_load']
        Parameters['valid_dl'] = loaders['Valid_load']
        Parameters['test_dl'] = loaders['Test_load']
        Parameters['num_classes'] = 45
        Parameters['matrix'] = M
        return Parameters

    elif type_con == "testing":
        ParamsTest = dict()
        ParamsTest['Test_Load'] = loaders['Test_load']
        ParamsTest['Models'] = Params
        ParamsTest['device'] = get_default_device()
        ParamsTest['dataset'] = dataset['Test']
        ParamsTest['dataset_targets'] = dataset['Test'].targets
        ParamsTest['n_classes'] = 45
        ParamsTest['Weights_None'] = True
        ParamsTest['batch_size'] = 16
        return ParamsTest


    else:
        print("Error: ~{}~ is not a valid type of configuration.!".format(type_con))




def test(args):
    DatasetParams = config(type_con='loaders')
    CD = CustomDatasets(DatasetParams)
    dataset = CD._load_Dataset()
    print("Cifar_10 dataset \n", dataset)

    loaders = CD.make_loaders()
    show_dataset(args, dataset=dataset['Train'])


    print("Cuda is available: ", torch.cuda.is_available())
    # load the train and validation into batches.
    ModelParams = config(type_con= 'models')


    #model = Net(n_channels = 3,n_classes = 10)
    #myvgg = MyVgg16(dataset_name='CIFAR10', trained=False)
    #model = myvgg.construct(n_classes=45)

    model = str('Cifar10-ResNet101')
    print
    Parameters = config(type_con='train', model=model, model2=None, Params=ModelParams, loaders=loaders, dataset=dataset)
    training_obg = BaseClassifier(Parameters)
    [tr_loss, tr_acc], [val_loss, val_acc] = training_obg.train_model(validation=True)
    test_loss, test_acc, y_pred, y_true = training_obg.test_model(model2 = None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Cifar-10 Test Set.", parents=[get_args_parser()])
    args = parser.parse_args()
    test(args)