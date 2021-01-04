from ..evaluation import lin_eval_testmoduls as let
from . import get_model as gm
from . import transformations as t 
from . import lars as lars
from . import dataloader as satelliteLoader
import numpy as np 
import torch 
import torchvision

#######################################
###### Global Variables ###############
#######################################
CIFAR_NUMWORKERS = 2
IMAGENET_NUMWORKERS = 8
SATELLITE_NUMWORKERS = 0

CIFAR_IMGSIZE = 32
IMAGENET_IMGSIZE = 128
SATELLITE_IMGSIZE = 224

# NOT USED IN DEFAULT SETTINGS
CIFAR_MEAN, CIFAR_STD  = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
IMAGENET_MEAN, IMAGENET_STD  = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] # mean/std for 224x224, HERE: 128 pixel used
SATELLITE_MEAN, SATELLITE_STD = [0.3712, 0.3683, 0.3598], [0.2535, 0.2431, 0.2412] # mean/std for 333x333, HERE: 224 pixel used

CIFAR_SPLIT = [5000, 5000]
SATELLITE_SPLIT = [209, 90]





##############################
###### Datasets ###############
###############################
def data_trainingSimclr(args):
    """

    """
    if args.dataset == "CIFAR10":
        dataset_train = torchvision.datasets.CIFAR10(args.dataset_dir, train = True, download=True, transform=t.TransformsSimCLR(imgsize=CIFAR_IMGSIZE, numberViews=args.numberviews))   
        
        if args.labels == "1%":
            train_X = np.load('data/sparse/cifar10/cifar_1%_xtrain.npy')
            train_y = np.load('data/sparse/cifar10/cifar_1%_xlabels.npy')
            dataset_train_nonaug = torch.utils.data.TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_y))
        
        elif args.labels == "10%":
            train_X = np.load('data/sparse/cifar10/cifar_10%_xtrain.npy')
            train_y = np.load('data/sparse/cifar10/cifar_10%_xlabels.npy')
            dataset_train_nonaug = torch.utils.data.TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_y))
        
        else: 
            dataset_train_nonaug = torchvision.datasets.CIFAR10(args.dataset_dir, train = True, download=True, transform=t.TransformsSimCLR(imgsize=CIFAR_IMGSIZE).test_transform) 
        
        dataset_test = torchvision.datasets.CIFAR10(args.dataset_dir, train = False, download=True, transform=t.TransformsSimCLR(imgsize=CIFAR_IMGSIZE).test_transform)
        _, dataset_validation = torch.utils.data.random_split(dataset_test, CIFAR_SPLIT)
        num_workers = CIFAR_NUMWORKERS

    elif args.dataset == "IMAGENET":
        
        dataset_train = torchvision.datasets.ImageFolder(args.dataset_dir+"/Data/train", transform=t.TransformsSimCLR(imgsize=IMAGENET_IMGSIZE, numberViews=args.numberviews))
        
        if args.labels == "1%":
            dataset_train_nonaug = torchvision.datasets.ImageFolder("data/sparse/imagenet224/1%/train", transform=t.TransformsSimCLR(imgsize=IMAGENET_IMGSIZE).test_transform)
        
        elif args.labels == "10%":
            dataset_train_nonaug = torchvision.datasets.ImageFolder("data/sparse/imagenet224/10%/train", transform=t.TransformsSimCLR(imgsize=IMAGENET_IMGSIZE).test_transform)
       
        else:
            dataset_train_nonaug = torchvision.datasets.ImageFolder(args.dataset_dir+"/Data/train", transform=t.TransformsSimCLR(imgsize=IMAGENET_IMGSIZE).test_transform)
        
        dataset_validation = torchvision.datasets.ImageFolder(args.dataset_dir+"/Data/val", transform=t.TransformsSimCLR(imgsize=IMAGENET_IMGSIZE).test_transform)
        num_workers = IMAGENET_NUMWORKERS

    elif args.dataset == "SATELLITE":
        dataset_train = satelliteLoader.SatelliteDataset(folder = "../../../data/bangalore/raster/strips_north_2016/", transform = t.TransformsSimCLR_SAT(imgsize=224, numberViews=args.numberviews), iterate_over_all=True)
        dataset_train_nonaug = satelliteLoader.SatelliteDatasetLabelled(transform=t.TransformsSimCLR_SAT(imgsize=224).test_transform_SAT_SUP, dset="train")
        dataset_validation = satelliteLoader.SatelliteDatasetLabelled(transform=t.TransformsSimCLR_SAT(imgsize=224).test_transform_SAT_SUP, dset="val")
        num_workers = SATELLITE_NUMWORKERS
    
    else:
        raise NotImplementedError

    trainloader, trainloader_nonaug, valloader = dataloaders_trainingSimclr(dataset_train, dataset_train_nonaug, dataset_validation, num_workers, args)

    return trainloader, trainloader_nonaug, valloader

def data_trainingSupervised(args):
    if args.dataset == "CIFAR10":

        if args.labels == "1%":
            train_X = np.load('data/sparse/cifar10/cifar_1%_xtrain_augmented.npy')
            train_y = np.load('data/sparse/cifar10/cifar_1%_xlabels_augmented.npy')
            dataset_train = torch.utils.data.TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_y))
        
        elif args.labels == "10%":
            train_X = np.load('data/sparse/cifar10/cifar_10%_xtrain_augmented.npy')
            train_y = np.load('data/sparse/cifar10/cifar_10%_xlabels_augmented.npy')
            dataset_train = torch.utils.data.TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_y))
        
        else:
            dataset_train = torchvision.datasets.CIFAR10(args.dataset_dir, train = True, download=True, transform=t.TransformsSimCLR(imgsize=CIFAR_IMGSIZE).train_transform) 
        
        dataset_test = torchvision.datasets.CIFAR10(args.dataset_dir, train = False, download=True, transform=t.TransformsSimCLR(imgsize=CIFAR_IMGSIZE).test_transform)
        _, dataset_validation = torch.utils.data.random_split(dataset_test, CIFAR_SPLIT)
        num_workers = CIFAR_NUMWORKERS

    elif args.dataset == "IMAGENET":
        if args.labels == "1%":
            dataset_train = torchvision.datasets.ImageFolder("data/sparse/imagenet/1%/train", transform=t.TransformsSimCLR(imgsize=IMAGENET_IMGSIZE).train_transform)
        
        elif args.labels == "10%":
            dataset_train = torchvision.datasets.ImageFolder("data/sparse/imagenet/10%/train", transform=t.TransformsSimCLR(imgsize=IMAGENET_IMGSIZE).train_transform)
        
        else:
            dataset_train = torchvision.datasets.ImageFolder(args.dataset_dir+"/Data/train", transform=t.TransformsSimCLR(imgsize=IMAGENET_IMGSIZE).train_transform)

        dataset_validation = torchvision.datasets.ImageFolder(args.dataset_dir+"/Data/val", transform=t.TransformsSimCLR(imgsize=IMAGENET_IMGSIZE).test_transform)
        num_workers = IMAGENET_NUMWORKERS
        
    elif args.dataset == "SATELLITE":
        dataset = satelliteLoader.SatelliteDatasetLabelled(transform=t.TransformsSimCLR_SAT(imgsize=224).train_transform_SAT_SUP)
        dataset_train = dataset 
        dataset.dataset, dataset.transform = "val", t.TransformsSimCLR_SAT(imgsize=224).test_transform_SAT_SUP
        dataset_validation = dataset 
        num_workers = SATELLITE_NUMWORKERS
    else:
        raise NotImplementedError

    trainloader, valloader = dataloaders_trainingSupervised(dataset_train, dataset_validation, num_workers, args)

    return trainloader, valloader

##############################
###### Dataloaders ###############
###############################
def dataloaders_trainingSimclr(dataset_train, dataset_train_nonaug, dataset_validation, num_workers, args):
    """

    """
    trainloader_nonaug = torch.utils.data.DataLoader(dataset_train_nonaug, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    valloader = torch.utils.data.DataLoader(dataset_validation, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    return trainloader, trainloader_nonaug, valloader

def dataloaders_trainingSupervised(dataset_train, dataset_validation, num_workers, args):
    """

    """
    trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    valloader = torch.utils.data.DataLoader(dataset_validation, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    return trainloader, valloader

##############################
###### OTHER ###############
###############################
def optimizer_training(model, args):
    """

    """
    if str.lower(args.optimizer) == "lars": optimizer = lars.LARS(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9, max_epoch=args.epochs, warmup_epochs=round(0.1*args.epochs))
    elif str.lower(args.optimizer) == "sgd": optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)
    elif str.lower(args.optimizer) == "adamw": optimizer = torch.optim.AdamW(model.parameters(), args.lr)
    else: optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    return optimizer

def encoder(args, pretrained=False):
    """

    """
    if args.dataset == "CIFAR10":
        encoder = gm.modify_resnet_model(gm.get_resnet(args.arch, pretrained=pretrained))
    elif args.dataset == "IMAGENET":
        encoder = gm.get_resnet(args.arch, pretrained=pretrained)
    else:
        encoder = gm.modify_resnet_model_SATELLITE(gm.get_resnet(args.arch, pretrained=pretrained))

    return encoder

def numberClasses(args):
    """

    """
    if args.dataset == "CIFAR10":
        outputdim = 10
    elif args.dataset == "IMAGENET":
        outputdim = 1000
    else:
        outputdim = 2

    return outputdim