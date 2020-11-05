# Import own modules
# SelfSupervised
import modules.selfsupervised.simclr_model as sm 
import modules.selfsupervised.simclr_train as smtr
# Supervised
import modules.supervised.supervised_model as supm
import modules.supervised.supervised_train as st
# Finetuning
import modules.finetuning.finetune_model as finm
# Evaluation
import modules.evaluation.lin_eval_testmoduls as let
# Utils
import modules.utils.get_model as gm
import modules.utils.transformations as t 
import modules.utils.loss_functions as lf
import modules.utils.lars as lars
import modules.utils.imagenet as inet 
# Import extern modules
import argparse
import torchvision
import torch 
import torch.nn as nn
from tqdm import tqdm
import math 
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np

# Create an Argparser
parser = argparse.ArgumentParser(description='Train simCLRv1 on CIFAR-10')
# Model-Architecture
parser.add_argument('-a', '--arch', default='resnet18')
# lr: 0.06 for batch 512 (or 0.03 for batch 256) 0.12 for 1024?
parser.add_argument('--lr', '--learning-rate', default=0.0006, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--epochs', default=20, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int, help='learning rate schedule (when to drop lr by 10x); does not take effect if --cos is on')
parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')
parser.add_argument('--batch-size', default=256, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--wd', default=1e-4, type=float, metavar='W', help='weight decay')
parser.add_argument('--dataset', default="IMAGENET", type=str, metavar='W', help='dataset')
parser.add_argument('--dataset-dir', default="/data", type=str, metavar='W', help='dataset directory')
parser.add_argument('--method', default="selfsupervised", type=str, metavar='W', help='Selfsupervised, Supervised, Finetuned')
parser.add_argument('--optimizer', default="adam", type=str, metavar='W', help='Adam, SGD, Lars')
# SIMCLR specific configs:
parser.add_argument('--dim', default=512, type=int, help='feature dimension')
parser.add_argument('--t', default=0.2, type=float, help='softmax temperature')
parser.add_argument('--numberviews', default=2, type=int, help='Number of Views created for Avg.Loss')
# Training/Test Settings
parser.add_argument('--train', default="True", type=str, metavar='W', help='training true or false')
parser.add_argument('--test', default="True", type=str, metavar='W', help='test true or false')
parser.add_argument('--model-dir', default="", type=str, metavar='W', help='if test true give model dir!')
# Linear Evaluation
parser.add_argument('--epochs-lineval', default=500, type=int, metavar='N', help='number of total epochs to run in linEval')
# Labels
parser.add_argument('--labels', default="full", type=str, metavar='N', help='1%, 10% or full labelled data sets are supported')
# Set up args
args = parser.parse_args()

# Convert 
args.train = True if str.lower(args.train) == 'true' else False
args.test = True if str.lower(args.test) == 'true' else False
# Current Date
now = datetime.now()
dt_string = now.strftime("%d.%m.%Y_%H")
# Create a logger 
if args.train != True and args.test != True: raise NotImplementedError
if args.train == False and args.test == True:
    if args.model_dir == "": raise NotImplementedError
    writer = SummaryWriter()
else: writer = SummaryWriter(log_dir="runs/simclr"+str(dt_string)+"_model"+str(args.arch)+"_lr"+str(args.lr)+"_epochs"+str(args.epochs)+"_batch_size"+str(args.batch_size)+"_dim"+str(args.dim)+"_t"+str(args.t)+"_data"+str(args.dataset))

def training_model(): 
    # Datasets
    if args.dataset == "CIFAR10":
        if str.lower(args.method) == "supervised" or str.lower(args.method) == "finetuned":
            if args.labels == "1%":
                train_X = np.load('data/cifar_1%_xtrain_augmented.npy')
                train_y = np.load('data/cifar_1%_xlabels_augmented.npy')
                dataset_train = torch.utils.data.TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_y))
                train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
            elif args.labels == "10%":
                train_X = np.load('data/cifar_10%_xtrain_augmented.npy')
                train_y = np.load('data/cifar_10%_xlabels_augmented.npy')
                dataset_train = torch.utils.data.TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_y))
                train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
            else:
                dataset_train = torchvision.datasets.CIFAR10(args.dataset_dir, train = True, download=True, transform=t.TransformsSimCLR(numberViews=args.numberviews))   
                train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
        else:
            dataset_train = torchvision.datasets.CIFAR10(args.dataset_dir, train = True, download=True, transform=t.TransformsSimCLR(numberViews=args.numberviews))   
            train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
        encoder = gm.modify_resnet_model(gm.get_resnet(args.arch, pretrained=False))
        outputdim = 10
    elif args.dataset == "IMAGENET":
        data_path = "C:/Users/Dustin/Desktop/simclrv1_plus_v2/imagenet_new/Data/train"
        dataset_train = inet.get_imagenet_datasets(data_path, test=False, numberViews=args.numberviews)
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle = True, num_workers=4, drop_last=True)
        encoder = gm.get_resnet(args.arch, pretrained=False)
        outputdim = 1000
    else:
        raise NotImplementedError

    # Number of Features
    n_features = encoder.fc.in_features  # get dimensions of fc layer

    if str.lower(args.method) == "supervised":
        model, criterion, epoch_start = supm.modelSupervised(encoder, n_features, outputdim).cuda(), nn.CrossEntropyLoss(), 1
        # Optimizer
        if str.lower(args.optimizer) == "lars": optimizer = lars.LARS(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)
        elif str.lower(args.optimizer) == "sgd": optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)
        else: optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        for epoch in range(epoch_start, args.epochs + 1):
            train_loss = st.train(model, train_loader, optimizer, criterion, epoch, args)
            writer.add_scalar("Loss/train_supervised", train_loss, epoch)
            if (epoch % 100) == 0: torch.save(model, "saved_models/supervised/"+str(dt_string)+"STEP_model"+str(args.arch)+"_lr"+str(args.lr)+"_epochs"+str(100)+"_batch_size"+str(args.batch_size)+"_data"+str(args.dataset)+".pth")
        writer.flush()
        torch.save(model, "saved_models/supervised/"+str(dt_string)+"_model"+str(args.arch)+"_lr"+str(args.lr)+"_epochs"+str(args.epochs)+"_batch_size"+str(args.batch_size)+"_data"+str(args.dataset)+".pth")
        return model
    elif str.lower(args.method) == "finetuned":
        model = torch.load(args.model_dir)
        encoder = model.encoder_f
        n_features = model.n_features 
        model, epoch_start = finm.modelSupervised(encoder,n_features, outputdim).cuda(), 1
        # Optimizer
        if str.lower(args.optimizer) == "lars": optimizer = lars.LARS(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)
        elif str.lower(args.optimizer) == "sgd": optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)
        else: optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(epoch_start, args.epochs + 1):
            train_loss = st.train(model, train_loader, optimizer, criterion, epoch, args)
            writer.add_scalar("Loss/train_finetuned", train_loss, epoch)
        writer.flush()
        torch.save(model, "saved_models/finetuned/"+str(dt_string)+"_model"+str(args.arch)+"_lr"+str(args.lr)+"_epochs"+str(args.epochs)+"_batch_size"+str(args.batch_size)+"_data"+str(args.dataset)+".pth")
        return model
    else:
        model, epoch_start = sm.modelSIMCLR(encoder, n_features, lf.contrastive_loss_cosine_extra, dim=args.dim, T=args.t).cuda(), 1
        # Choose Optimizer
        if str.lower(args.optimizer) == "lars": optimizer = lars.LARS(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)
        elif str.lower(args.optimizer) == "sgd": optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)
        else: optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        # Start Training
        for epoch in range(epoch_start, args.epochs + 1):
            train_loss, neg_sim, pos_sim = smtr.train(model, train_loader, optimizer, epoch, args)
            writer.add_scalar("Loss/train_simCLR", train_loss, epoch)
            writer.add_scalar("Positive_SIM/train_simCLR", pos_sim, epoch)
            writer.add_scalar("Negative_SIM/train_simCLR", neg_sim, epoch)
            if (epoch % 100) == 0: torch.save(model, "saved_models/simclr/"+str(dt_string)+"STEP_model"+str(args.arch)+"_lr"+str(args.lr)+"_epochs"+str(epoch)+"_batch_size"+str(args.batch_size)+"_dim"+str(args.dim)+"_t"+str(args.t)+"_data"+str(args.dataset)+".pth")
        writer.flush()
        torch.save(model, "saved_models/simclr/"+str(dt_string)+"_model"+str(args.arch)+"_lr"+str(args.lr)+"_epochs"+str(args.epochs)+"_batch_size"+str(args.batch_size)+"_dim"+str(args.dim)+"_t"+str(args.t)+"_data"+str(args.dataset)+".pth")
        return model

def test_model(model=None): 
    if args.train == False and args.test == True: 
        model = torch.load(args.model_dir)

    if args.dataset == "CIFAR10":
        if args.labels == "1%":
            train_X = np.load('data/cifar_1%_xtrain.npy')
            train_y = np.load('data/cifar_1%_xlabels.npy')
            dataset_train = torch.utils.data.TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_y))
            dataset_test = torchvision.datasets.CIFAR10(args.dataset_dir, train = False, download=True, transform=t.TransformsSimCLR().test_transform)
        elif args.labels == "10%":
            train_X = np.load('data/cifar_10%_xtrain.npy')
            train_y = np.load('data/cifar_10%_xlabels.npy')
            dataset_train = torch.utils.data.TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_y))
            dataset_test = torchvision.datasets.CIFAR10(args.dataset_dir, train = False, download=True, transform=t.TransformsSimCLR().test_transform)
        else:
            dataset_train = torchvision.datasets.CIFAR10(args.dataset_dir, train = True, download=True, transform=t.TransformsSimCLR().test_transform)
            dataset_test = torchvision.datasets.CIFAR10(args.dataset_dir, train = False, download=True, transform=t.TransformsSimCLR().test_transform)
        memory_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=0)
        linear_model = let.LogisticRegression(model.n_features, 10).cuda()
    elif args.dataset == "IMAGENET":
        data_path = "C:/Users/Dustin/Desktop/simclrv1_plus_v2/imagenet_new/Data/train"
        dataset_train, dataset_test = inet.get_imagenet_datasets(data_path, check_transform=False)
        memory_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=4)
        linear_model = let.LogisticRegression(model.n_features, 1000).cuda()
    else:
        raise NotImplementedError

    # Get Encoder
    encoder_f = model.encoder_f
    # Create Training and Test-Set for Regression Model
    (train_X, train_y, test_X, test_y) = let.get_features(encoder_f, memory_loader, test_loader)
    arr_train_loader, arr_test_loader = let.create_data_loaders_from_arrays(train_X, train_y, test_X, test_y, args.batch_size)
    # Set up training
    optimizer = torch.optim.Adam(linear_model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()
    # Training
    for epoch in range(args.epochs_lineval):
        loss_epoch, accuracy_epoch = let.train_lr(arr_train_loader, linear_model, criterion, optimizer, epoch, args)
        writer.add_scalar("Loss/train_linEval", loss_epoch / len(memory_loader), epoch)
        writer.add_scalar("Accuracy/train_linEval", accuracy_epoch / len(memory_loader), epoch)
    # final testing
    loss_epoch, accuracy_epoch = let.test_lr(arr_test_loader, linear_model, criterion, optimizer)
    writer.add_scalar("Loss/test_linEval", loss_epoch / len(test_loader), epoch)
    writer.add_scalar("Accuracy/test_linEval", accuracy_epoch / len(test_loader), epoch)
    writer.flush()
    print(f"[FINAL]\t Loss: {loss_epoch / len(test_loader)}\t Accuracy: {accuracy_epoch / len(test_loader)}")

if __name__ == '__main__':
    if args.train == True and args.test == False:
        model = training_model()
    if args.train == True and args.test == True:
        model = training_model()
        test_model(model)
    if args.train == False and args.test == True:
        test_model()