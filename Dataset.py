
"""
Created on 23/10/2022

@author: Dimitrios Papaioannou
"""

from torch.utils.data.dataloader import DataLoader
import torchvision

from torchvision import transforms
from Plots import *
from torch.utils.data.sampler import SubsetRandomSampler

class CustomDatasets(object):
    """
    MnistMultiLoaders is a class which is providing a manipulation for the Mnist Dataset.
    - Requires as input a dictionary which will set the nessecary infos for the creation of Mnist loaders
    (e.g., transform ~boolean~, path, batch size ~int~).
    - Target_type specifies if we want the labels to be in the default mode or in a one hot encoding formation. Require values:
      - "Ohe" - One hot Encoding.
      - "-" - for Default formation.
    - The argument nums serve the creation of the Custom Loader, requires a list as input and by default is set to be a loader with
    only target labels 1, 2, 3.
    """

    def __init__(self, args, transform_matrix):
        self.path = args.path
        self.target_type = args.target_type
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.n_classes = args.n_classes
        self.transform_img = args.transform
        self.validation = args.validation
        self.random_seed = args.Random_Seed
        self.valid_size = args.Valid_Size
        self.shuffle = args.shuffle
        self.pin_memory = args.pin_memory
        self.num_workers = args.n_workers
        self.matrix = transform_matrix
        self.dataset = None

    def _load_transforms(self):
        if self.dataset_name == "MNIST" or self.dataset_name == "F-MNIST":
            if self.transform_img:
                transform_train = torchvision.transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(size=29),
                    # transforms.CenterCrop(size=28),
                    # transforms.Grayscale(num_output_channels=3),
                    transforms.Normalize((0.485,), (0.229,))
                ])

                transform_valid = torchvision.transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(size=29),
                    # transforms.CenterCrop(size=28),
                    # transforms.Grayscale(num_output_channels=3),
                    transforms.Normalize((0.485,), (0.229,))
                ])
                transform_test = torchvision.transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(size=29),
                    # transforms.CenterCrop(size=28),
                    # transforms.Grayscale(num_output_channels=3),
                    transforms.Normalize((0.485,), (0.229,))
                ])
            else:
                transform_train = None
                transform_valid = None
                transform_test = None

        elif self.dataset_name == "CIFAR10" or self.dataset_name == "CIFAR100":
            if self.transform_img:
                transform_train = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(10),
                    transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                ])

                transform_valid = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                ])

                transform_test = transforms.Compose([
                    transforms.Resize((32,32)),
                    # transforms.RandomCrop(32, padding=4),
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
                ])
            else:
                transform_train = None
                transform_valid = None
                transform_test = None

        elif self.dataset_name == "GTSRB":
            if self.transform_img:
                transform_train = transforms.Compose([
                    transforms.Resize((96, 96)),
                    transforms.RandomCrop(96, padding=4),
                    transforms.RandomHorizontalFlip(1),
                    transforms.RandomRotation(10),
                    # transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                    # transforms.ColorJitter(brightness=2, contrast=0.5, saturation=0.5, hue=0.4),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629]),
                ])

                transform_valid = transforms.Compose([
                    transforms.Resize((96, 96)),
                    transforms.RandomCrop(96, padding=4),
                    transforms.RandomHorizontalFlip(1),
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629]),
                ])

                transform_test = transforms.Compose([
                    transforms.Resize((96, 96)),
                    transforms.RandomCrop(96, padding=4),
                    transforms.RandomHorizontalFlip(1),
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629]),
                ])
            else:
                transform_train = None
                transform_valid = None
                transform_test = None

        elif self.dataset_name == "SVHN":
            if self.transform_img:
                transform_train = transforms.Compose([
                    transforms.Resize((94, 94)),
                    transforms.RandomCrop(94, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(10),
                    transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean = [0.4376821, 0.4437697, 0.47280442], std = [0.19803012, 0.20101562, 0.19703614]),
                ])

                transform_valid = transforms.Compose([
                    transforms.Resize((94, 94)),
                    transforms.RandomCrop(94, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean = [0.4376821, 0.4437697, 0.47280442], std = [0.19803012, 0.20101562, 0.19703614]),
                ])

                transform_test = transforms.Compose([
                    transforms.Resize((94, 94)),
                    transforms.RandomCrop(94, padding=4),
                    transforms.ToTensor(),
                    # transforms.Normalize(mean = [0.4376821, 0.4437697, 0.47280442], std = [0.19803012, 0.20101562, 0.19703614]),
                ])
            else:
                transform_train = None
                transform_valid = None
                transform_test = None

        elif self.dataset_name == "STL-10":
            if self.transform_img:
                transform_train = transforms.Compose([
                    transforms.Resize((96, 96)),
                    transforms.RandomCrop(96, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(10),
                    # transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.4272, 0.4200, 0.3885], std=[0.2395, 0.2363, 0.2357]),
                ])

                transform_valid = transforms.Compose([
                    transforms.Resize((96, 96)),
                    transforms.RandomCrop(96, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(10),
                    # transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.4272, 0.4200, 0.3885], std=[0.2395, 0.2363, 0.2357]),
                ])

                transform_test = transforms.Compose([
                    transforms.Resize((96, 96)),
                    # transforms.RandomCrop(96, padding=4),
                    # transforms.RandomHorizontalFlip(1),
                    # transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=[0.4272, 0.4200, 0.3885], std=[0.2395, 0.2363, 0.2357]),
                ])
            else:
                transform_train = None
                transform_valid = None
                transform_test = None

        elif self.dataset_name == "CALTECH101" or self.dataset_name == "CALTECH256":
            if self.transform_img:
                transform_train = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                transform_valid = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

                transform_test = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
            else:
                transform_train = None
                transform_valid = None
                transform_test = None

        elif self.dataset_name == "DTD":
            if self.transform_img:
                transform_train = transforms.Compose([
                    transforms.RandomCrop(96, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5323, 0.4749, 0.4255], std=[0.1568, 0.1584, 0.1546]),
                ])
                transform_valid = transforms.Compose([
                    transforms.RandomCrop(96, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5323, 0.4749, 0.4255], std=[0.1568, 0.1584, 0.1546]),
                ])

                transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5323, 0.4749, 0.4255], std=[0.1568, 0.1584, 0.1546]),
                ])
            else:
                transform_train = None
                transform_valid = None
                transform_test = None
        else:
            msg = 'Invalid name of dataset. Received {} but expected "MNIST", "F-MNIST", "CIFAR10" or "CIFAR100"'
            raise ValueError(msg.format(self.dataset_name))

        if self.target_type == 'OHE':
            target_trans = (
                lambda y: torch.zeros(self.n_classes, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
        elif self.target_type == "OVO":
            target_trans = (lambda y: torch.sum((torch.Tensor(torch.zeros(self.n_classes, dtype=torch.float).scatter_(0, torch.tensor(y),value=1)) * self.matrix).T, axis=0))
        else:
            target_trans = None

        return transform_train, transform_valid, transform_test, target_trans

    def _load_Dataset(self):
        transform_train, transform_valid, transform_test, target_trans = self._load_transforms()

        if self.dataset_name == "MNIST":
            if self.validation:
                self.dataset = {
                    "Train": torchvision.datasets.MNIST(root=self.path, transform=transform_train, download=True,
                                                        target_transform=target_trans),
                    "Valid": torchvision.datasets.MNIST(root=self.path, transform=transform_valid, download=True,
                                                        target_transform=target_trans),
                    "Test": torchvision.datasets.MNIST(root=self.path, train=False, transform=transform_test,
                                                       download=True, target_transform=target_trans)}
            else:
                self.dataset = {
                    "Train": torchvision.datasets.MNIST(root=self.path, transform=transform_train, download=True,
                                                        target_transform=target_trans),
                    "Test": torchvision.datasets.MNIST(root=self.path, train=False, transform=transform_test,
                                                       download=True, target_transform=target_trans)}

        elif self.dataset_name == "F-MNIST":
            if self.validation:
                self.dataset = {
                    "Train": torchvision.datasets.FashionMNIST(root=self.path, transform=transform_train, download=True,
                                                               target_transform=target_trans),
                    "Valid": torchvision.datasets.FashionMNIST(root=self.path, transform=transform_valid, download=True,
                                                               target_transform=target_trans),
                    "Test": torchvision.datasets.FashionMNIST(root=self.path, train=False, transform=transform_test,
                                                              download=True, target_transform=target_trans)}
            else:
                self.dataset = {
                    "Train": torchvision.datasets.FashionMNIST(root=self.path, transform=transform_train, download=True,
                                                               target_transform=target_trans),
                    "Test": torchvision.datasets.FashionMNIST(root=self.path, train=False, transform=transform_test,
                                                              download=True, target_transform=target_trans)}
        elif self.dataset_name == "CIFAR10":
            if self.validation:
                self.dataset = {
                    "Train": torchvision.datasets.CIFAR10(root=self.path, transform=transform_train, download=True,
                                                          target_transform=target_trans),
                    "Valid": torchvision.datasets.CIFAR10(root=self.path, transform=transform_valid, download=True,
                                                          target_transform=target_trans),
                    "Test": torchvision.datasets.CIFAR10(root=self.path, train=False, transform=transform_test,
                                                         download=True, target_transform=target_trans)}
            else:

                self.dataset = {
                    "Train": torchvision.datasets.CIFAR10(root=self.path, transform=transform_train, download=True,
                                                          target_transform=target_trans, ),
                    "Test": torchvision.datasets.CIFAR10(root=self.path, train=False, transform=transform_test,
                                                         download=True, target_transform=target_trans)}

        elif self.dataset_name == "SVHN":
            if self.validation:
                self.dataset = {
                    "Train": torchvision.datasets.SVHN(root=self.path, transform=transform_train, split="train",
                                                       download=True, target_transform=target_trans),
                    "Valid": torchvision.datasets.SVHN(root=self.path, transform=transform_valid, split='extra',
                                                       download=True, target_transform=target_trans),
                    "Test": torchvision.datasets.SVHN(root=self.path, split='test', transform=transform_test,
                                                      download=True, target_transform=target_trans)}
            else:
                self.dataset = {
                    "Train": torchvision.datasets.SVHN(root=self.path, transform=transform_train, split="train",
                                                       download=True, target_transform=target_trans),
                    "Test": torchvision.datasets.SVHN(root=self.path, split='test', transform=transform_test,
                                                      download=True, target_transform=target_trans)}
        elif self.dataset_name == "GTSRB":
            if self.validation:
                self.dataset = {
                    "Train": torchvision.datasets.GTSRB(root=self.path, transform=transform_train, split="train",
                                                       download=True, target_transform=target_trans),
                    "Valid": torchvision.datasets.GTSRB(root=self.path, transform=transform_valid, split='train',
                                                       download=True, target_transform=target_trans),
                    "Test": torchvision.datasets.GTSRB(root=self.path, split='test', transform=transform_test,
                                                      download=True, target_transform=target_trans)}
            else:
                self.dataset = {
                    "Train": torchvision.datasets.GTSRB(root=self.path, transform=transform_train, split="train",
                                                       download=True, target_transform=target_trans),
                    "Test": torchvision.datasets.GTSRB(root=self.path, split='test', transform=transform_test,
                                                      download=True, target_transform=target_trans)}
        elif self.dataset_name == "STL-10":
            if self.validation:
                self.dataset = {
                    "Train": torchvision.datasets.STL10(root=self.path, transform=transform_train, split="train",
                                                        download=True, target_transform=target_trans),
                    "Valid": torchvision.datasets.STL10(root=self.path, transform=transform_valid, split="test",
                                                        download=True, target_transform=target_trans),
                    "Test": torchvision.datasets.STL10(root=self.path, split='test', transform=transform_test,
                                                       download=True, target_transform=target_trans)}
            else:
                self.dataset = {
                    "Train": torchvision.datasets.STL10(root=self.path, transform=transform_train, split="train",
                                                        download=True, target_transform=target_trans),
                    "Test": torchvision.datasets.STL10(root=self.path, split='test', transform=transform_test,
                                                       download=True, target_transform=target_trans)}

        elif self.dataset_name == "DTD":
            if self.validation:
                self.dataset = {
                    "Train": torchvision.datasets.DTD(root=self.path, transform=transform_train, split="train",
                                                    download=True, target_transform=target_trans),
                    "Valid": torchvision.datasets.DTD(root=self.path, transform=transform_valid, split="val",
                                                        download=True, target_transform=target_trans),
                    "Test": torchvision.datasets.DTD(root=self.path, split='test', transform=transform_test,
                                                       download=True, target_transform=target_trans)}
            else:
                self.dataset = {
                    "Train": torchvision.datasets.STL10(root=self.path, transform=transform_train, split="train",
                                                        download=True, target_transform=target_trans),
                    "Test": torchvision.datasets.STL10(root=self.path, split='test', transform=transform_test,
                                                       download=True, target_transform=target_trans)}

        elif self.dataset_name == "CIFAR100":
            if self.validation:
                self.dataset = {
                    "Train": torchvision.datasets.CIFAR100(root=self.path, transform=transform_train, download=True,
                                                           target_transform=target_trans),
                    "Valid": torchvision.datasets.CIFAR100(root=self.path, transform=transform_valid, download=True,
                                                           target_transform=target_trans),
                    "Test": torchvision.datasets.CIFAR100(root=self.path, train=False, transform=transform_test,
                                                          download=True, target_transform=target_trans)}
            else:
                self.dataset = {
                    "Train": torchvision.datasets.CIFAR100(root=self.path, transform=transform_train, download=True,
                                                           target_transform=target_trans),
                    "Test": torchvision.datasets.CIFAR100(root=self.path, train=False, transform=transform_test,
                                                          download=True, target_transform=target_trans)}

        elif self.dataset_name == "CALTECH256":
            if self.validation:
                self.dataset = {
                    "Train": torchvision.datasets.Caltech256(root=self.path, transform=transform_train, download=True,
                                                             target_transform=target_trans),
                    "Valid": torchvision.datasets.Caltech256(root=self.path, transform=transform_valid, download=True,
                                                             target_transform=target_trans),
                    "Test": torchvision.datasets.Caltech256(root=self.path, transform=transform_test,
                                                            download=True, target_transform=target_trans)}
            else:
                self.dataset = {
                    "Train": torchvision.datasets.Caltech256(root=self.path, transform=transform_train, download=True,
                                                             target_transform=target_trans),
                    "Test": torchvision.datasets.Caltech256(root=self.path, transform=transform_test,
                                                            download=True, target_transform=target_trans)}
        else:
            self.dataset = None
            print("Error: {} dataset is not Available.!".format(self.dataset_name))
        return self.dataset

    def make_loaders(self):
        """
        The parameter dataset_type require arguments of the form:
        - "Even" if you want to create an Even Dataloader
        - "Odd" if you want to create an Odd Dataloader
        - "Original" if you want to create the original DataLoader
        - "Custom" if you want to create a custom Dataloader

        Default is set to be the Original Dataloader

        """
        if self.dataset is None:
            self.dataset = self._load_dataset()

        if self.dataset_name == "SVHN" or self.dataset_name == "STL10" or self.dataset_name == "DTD":
            loader = {
                'Train_load': DataLoader(dataset=self.dataset['Train'], batch_size=self.batch_size,
                                         pin_memory=self.pin_memory, num_workers=self.num_workers),
                'Valid_load': DataLoader(dataset=self.dataset['Valid'], batch_size=self.batch_size,
                                         pin_memory=self.pin_memory, num_workers=self.num_workers),
                'Test_load': DataLoader(dataset=self.dataset['Test'], batch_size=self.batch_size,
                                        pin_memory=self.pin_memory, num_workers=self.num_workers)}
        else:
            num_train = len(self.dataset['Train'])
            indices = list(range(num_train))
            split = int(np.floor(self.valid_size * num_train))

            if self.shuffle:
                np.random.seed(self.random_seed)
                np.random.shuffle(indices)

            train_idx, valid_idx = indices[split:], indices[:split]
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)

            loader = {
                'Train_load': DataLoader(dataset=self.dataset['Train'], batch_size=self.batch_size,
                                         sampler=train_sampler, pin_memory=self.pin_memory, num_workers=self.num_workers),
                'Valid_load': DataLoader(dataset=self.dataset['Valid'], batch_size=self.batch_size,
                                         sampler=valid_sampler, pin_memory=self.pin_memory, num_workers=self.num_workers),
                'Test_load': DataLoader(dataset=self.dataset['Test'], batch_size=self.batch_size,
                                        pin_memory=self.pin_memory, num_workers=self.num_workers)}
        return loader