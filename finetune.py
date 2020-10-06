
import modules.supervised_model as supm
import modules.simclr_model as sm 
import modules.transformations as t 
import modules.supervised_train as st
import modules.lin_eval_testmoduls as let
#from modules.transformations import TransformsSimCLR
import argparse
import torchvision
import torchvision.models as models
import torch.nn as nn
from tqdm import tqdm
import math 
import torch 
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np

# Create an Argparser
parser = argparse.ArgumentParser(description='Train simCLRv1 on CIFAR-10')
# Model-Architecture
parser.add_argument('-a', '--arch', default='resnet18')
# lr: 0.06 for batch 512 (or 0.03 for batch 256) 0.12 for 1024?
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--epochs', default=20, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int, help='learning rate schedule (when to drop lr by 10x); does not take effect if --cos is on')
parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')
parser.add_argument('--batch-size', default=256, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--wd', default=5e-4, type=float, metavar='W', help='weight decay')
parser.add_argument('--dataset', default="CIFAR10", type=str, metavar='W', help='dataset')
parser.add_argument('--dataset-dir', default="/data", type=str, metavar='W', help='dataset directory')
#
parser.add_argument('--train', default="True", type=str, metavar='W', help='training true or false')
parser.add_argument('--test', default="True", type=str, metavar='W', help='test true or false')
parser.add_argument('--model-dir', default="", type=str, metavar='W', help='if test true give model dir!')
# Linear Evaluation
parser.add_argument('--epochs-lineval', default=500, type=int, metavar='N', help='number of total epochs to run in linEval')
# 
parser.add_argument('--labels', default="full", type=str, metavar='N', help='1%, 10% or full labelled data sets are supported')

args = parser.parse_args()

if args.train == "True":
    args.train = True
else:
    args.train = False

if args.test == "True":
    args.test = True
else: 
    args.test = False

# Current Date
now = datetime.now()
dt_string = now.strftime("%d.%m.%Y_%H")

# Create a logger 
if args.train != True and args.test != True:
    raise NotImplementedError

if args.model_dir == "":
        raise NotImplementedError

if args.train == False and args.test == True:
    writer = SummaryWriter()
else: 
    writer = SummaryWriter(log_dir="runs/finetuned"+str(dt_string)+"_model"+str(args.arch)+"_lr"+str(args.lr)+"_epochs"+str(args.epochs)+"_batch_size"
                                    +str(args.batch_size)+"_data"+str(args.dataset))

def training_model(): 
    if args.dataset == "CIFAR10":
        train_data = torchvision.datasets.CIFAR10(
        args.dataset_dir,
        train = True,
        download=True,
        transform=t.TransformsSimCLR(),
        )   
        test_data = torchvision.datasets.CIFAR10(
        args.dataset_dir,
        train = False,
        download=True,
        transform=t.TransformsSimCLR().test_transform,
        )
    else:
        raise NotImplementedError

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    if args.labels == "1%":
        train_X = np.load('data/cifar_1%_xtrain_augmented.npy')
        train_y = np.load('data/cifar_1%_xlabels_augmented.npy')
        train = torch.utils.data.TensorDataset(
        torch.from_numpy(train_X), torch.from_numpy(train_y)
        )
        train_loader = torch.utils.data.DataLoader(
            train, batch_size=args.batch_size, shuffle=True, drop_last=True
        )

    if args.labels == "10%":
        train_X = np.load('data/cifar_10%_xtrain_augmented.npy')
        train_y = np.load('data/cifar_10%_xlabels_augmented.npy')
        train = torch.utils.data.TensorDataset(
            torch.from_numpy(train_X), torch.from_numpy(train_y)
            )
        train_loader = torch.utils.data.DataLoader(
            train, batch_size=args.batch_size, shuffle=True
            )

    model = torch.load(args.model_dir)
    encoder = model.encoder_f
    n_features = model.n_features  # get dimensions of fc layer
    model = supm.modelSupervised(
        encoder,
        n_features,
    ).cuda()

    # define optimizer
    #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, nesterov=True, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    epoch_start = 1
    for epoch in range(epoch_start, args.epochs + 1):
        train_loss = st.train(model, train_loader, optimizer, criterion, epoch, args)
        writer.add_scalar("Loss/train_finetuned", train_loss, epoch)
        model.trainloss.append(train_loss)
        """
        if epoch == 100: 
            torch.save(model, "saved_models/finetuned/"+str(dt_string)+"STEP_model"+str(args.arch)+"_lr"+str(args.lr)+"_epochs"+str(100)+"_batch_size"+str(args.batch_size)
                                         +"_data"+str(args.dataset)+".pth")
        if epoch == 200: 
            torch.save(model, "saved_models/finetuned/"+str(dt_string)+"STEP_model"+str(args.arch)+"_lr"+str(args.lr)+"_epochs"+str(200)+"_batch_size"+str(args.batch_size)
                                         +"_data"+str(args.dataset)+".pth")
        """

    writer.flush()
    torch.save(model, "saved_models/finetuned/"+str(dt_string)+"_model"+str(args.arch)+"_lr"+str(args.lr)+"_epochs"+str(args.epochs)+"_batch_size"+str(args.batch_size)
                                                +"_data"+str(args.dataset)+".pth")

    # final testing
    loss_epoch, accuracy_epoch = let.test_lr(test_loader, model, criterion, optimizer)
    writer.add_scalar("Loss/test_finetuned", loss_epoch / len(test_loader), epoch)
    writer.add_scalar("Accuracy/test_finetuned", accuracy_epoch / len(test_loader), epoch)
    writer.flush()
    print(
        f"[FINAL]\t Loss: {loss_epoch / len(test_loader)}\t Accuracy: {accuracy_epoch / len(test_loader)}"
    )

    return model

"""
def test_model(model=None): 

    if args.train == False and args.test == True: 
        model = torch.load(args.model_dir)

    if args.dataset == "CIFAR10":
        test_data = torchvision.datasets.CIFAR10(
        args.dataset_dir,
        train = False,
        download=True,
        transform=t.TransformsSimCLR().test_transform,
        )
    else:
        raise NotImplementedError

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
        
    # final testing
    loss_epoch, accuracy_epoch = let.test_lr(arr_test_loader, linear_model, criterion, optimizer)
    writer.add_scalar("Loss/test_finetuned", loss_epoch / len(test_loader), epoch)
    writer.add_scalar("Accuracy/test_finetuned", accuracy_epoch / len(test_loader), epoch)
    writer.flush()
    print(
        f"[FINAL]\t Loss: {loss_epoch / len(test_loader)}\t Accuracy: {accuracy_epoch / len(test_loader)}"
)
"""

if __name__ == '__main__':
    """
    if args.train == True and args.test == False:
        model = training_model()
    if args.train == True and args.test == True:
        model = training_model()
        test_model(model)
    if args.train == False and args.test == True:
        test_model()
    """
    model = training_model()