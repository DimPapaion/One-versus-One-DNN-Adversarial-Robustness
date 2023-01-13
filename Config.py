import argparse


def get_args_parser():
    parser = argparse.ArgumentParser(description="Arguments for setup OvO Classification", add_help=False)

    parser.add_argument("--exp_num", type=str, default="2", help="Number of experimental training.")
    # Config Dataset
    parser.add_argument("--path", type= str, default="./", help="Path for the local hosted datasets.")
    parser.add_argument("--transform", type=bool, default=True, help="If augmentations should be applied in images on the dataset.")
    parser.add_argument("--target_type", type= str, default=None,choices=[None,'Ohe'],
                        help="Transformation for the labels:'None', 'OHE' and 'OVO'.")
    parser.add_argument("--dataset_name", type= str, default="STL-10",
                        choices=['CIFAR10', 'CIFAR100','F-MNIST', 'MNIST','SVHN', 'STL-10'],
                        help="Name of the selected dataset.")

    parser.add_argument("--n_classes", type= int, default=10, help="Number of classes in the selected Dataset.")

    #Validation Dataset Set up.
    parser.add_argument("--validation", type=bool, default=True, help="If a validation dataset should be generated.")
    parser.add_argument("--Random_Seed", type=int, default=10, help="A random seed for picking random samples for validation.")
    parser.add_argument("--Valid_Size", type=float, default=0.2, help="Size of the validation dataset.")

    #Config DataLoaders setup
    parser.add_argument("--batch_size", type= int, default=32, help="Batch size for the DataLoader.")
    parser.add_argument("--pin_memory", type=bool, default=True, help="If pin memory should be active or not")
    parser.add_argument("--n_workers", type=int, default=0, help="Number of workers.")
    parser.add_argument("--shuffle", type=bool, default=True, help="Shuffle the dataset")

    # Config Training setup

    parser.add_argument("--is_trainable", type=bool, default=True, help="Training the Models")
    parser.add_argument("--epochs", type=int, default=400, help="Number of epochs")
    parser.add_argument("--model_name", type=str, default='Resnet_101', help="Name of the model.")
    parser.add_argument("--path_checkpoint", type=str, default='./', help="Name of the model.")
    parser.add_argument("--optimizer", type=str, default='SGD', choices=['ADAM', 'SGD'],
                        help="Select the desired optimizer")
    parser.add_argument("--lr", type=float, default=0.1, help="Set learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Set weight decay")
    parser.add_argument("--momentum", type=float, default=0.9, help="Set momentum")
    parser.add_argument('--schedule', type=int, nargs='+', default=[80, 220, 320],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')

    parser.add_argument("--is_lr_step_adjusted", type=bool, default=True, help="If lr should be adjusted step-wise or not")
    parser.add_argument('--gpu', type=str, default='0')  # gpu to be used

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--use-cpu', action='store_true')

    # Config Adv_attacks


    return parser

