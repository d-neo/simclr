
import modules.simclr_model as sm 
import modules.transformations as t 
import modules.simclr_train as smtr
import modules.lin_eval_testmoduls as let
import modules.loss_functions as lf
import modules.lars as lars
import modules.imagenet as inet 
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
parser.add_argument('--lr', '--learning-rate', default=0.0006, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--epochs', default=20, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int, help='learning rate schedule (when to drop lr by 10x); does not take effect if --cos is on')
parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')
parser.add_argument('--batch-size', default=256, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--wd', default=1e-4, type=float, metavar='W', help='weight decay')
parser.add_argument('--dataset', default="IMAGENET", type=str, metavar='W', help='dataset')
parser.add_argument('--dataset-dir', default="/data", type=str, metavar='W', help='dataset directory')
# SIMCLR specific configs:
parser.add_argument('--dim', default=512, type=int, help='feature dimension')
parser.add_argument('--t', default=0.2, type=float, help='softmax temperature')
#
parser.add_argument('--train', default="True", type=str, metavar='W', help='training true or false')
parser.add_argument('--test', default="True", type=str, metavar='W', help='test true or false')
parser.add_argument('--model-dir', default="", type=str, metavar='W', help='if test true give model dir!')
# Linear Evaluation
parser.add_argument('--epochs-lineval', default=500, type=int, metavar='N', help='number of total epochs to run in linEval')
# Labels
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

if args.train == False and args.test == True:
    if args.model_dir == "":
        raise NotImplementedError
    writer = SummaryWriter()
else: 
    writer = SummaryWriter(log_dir="runs/simclr"+str(dt_string)+"_model"+str(args.arch)+"_lr"+str(args.lr)+"_epochs"+str(args.epochs)+"_batch_size"+str(args.batch_size)
                                    +"_dim"+str(args.dim)+"_t"+str(args.t)+"_data"+str(args.dataset))


def get_resnet(name, pretrained=False):
    resnets = {
        "resnet18": models.resnet18(pretrained=pretrained),
        "resnet50": models.resnet50(pretrained=pretrained),
    }
    if name not in resnets.keys():
        raise KeyError(f"{name} is not a valid ResNet version")
    return resnets[name]

def modify_resnet_model(model):
    conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    nn.init.kaiming_normal_(conv1.weight, mode='fan_out', nonlinearity='relu')
    model.conv1 = conv1
    model.maxpool = nn.Identity()
    return model

def training_model(): 
    if args.dataset == "CIFAR10":
        train_data = torchvision.datasets.CIFAR10(
        args.dataset_dir,
        train = True,
        download=True,
        transform=t.TransformsSimCLR(),
        )   
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
        encoder = modify_resnet_model(get_resnet(args.arch, pretrained=False))
    elif args.dataset == "IMAGENET":
        data_path = "C:/Users/Dustin/Desktop/simclrv1_plus_v2/imagenet_new/Data/train"
        dataset_train = inet.get_imagenet_datasets(data_path, test=False)
        train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle = True, num_workers=4, drop_last=True)
        encoder = get_resnet(args.arch, pretrained=False)
    else:
        raise NotImplementedError

    n_features = encoder.fc.in_features  # get dimensions of fc layer
    model = sm.modelSIMCLR(
        encoder,
        n_features,
        lf.contrastive_loss_cosine_extra,
        dim=args.dim,
        T=args.t,
    ).cuda()

    # define optimizer
    #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    #optimizer = lars.LARS(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    epoch_start = 1
    for epoch in range(epoch_start, args.epochs + 1):
        train_loss, neg_sim, pos_sim = smtr.train(model, train_loader, optimizer, epoch, args)
        writer.add_scalar("Loss/train_simCLR", train_loss, epoch)
        writer.add_scalar("Positive_SIM/train_simCLR", pos_sim, epoch)
        writer.add_scalar("Negative_SIM/train_simCLR", neg_sim, epoch)
        #writer.add_scalar("lr_adj/train", lr, epoch)
        model.trainloss.append(train_loss)
        if epoch == 100: 
            torch.save(model, "saved_models/simclr/"+str(dt_string)+"STEP_model"+str(args.arch)+"_lr"+str(args.lr)+"_epochs"+str(100)+"_batch_size"+str(args.batch_size)
                                         +"_dim"+str(args.dim)+"_t"+str(args.t)+"_data"+str(args.dataset)+".pth")
        if epoch == 200: 
            torch.save(model, "saved_models/simclr/"+str(dt_string)+"STEP_model"+str(args.arch)+"_lr"+str(args.lr)+"_epochs"+str(200)+"_batch_size"+str(args.batch_size)
                                         +"_dim"+str(args.dim)+"_t"+str(args.t)+"_data"+str(args.dataset)+".pth")

    writer.flush()
    torch.save(model, "saved_models/simclr/"+str(dt_string)+"_model"+str(args.arch)+"_lr"+str(args.lr)+"_epochs"+str(args.epochs)+"_batch_size"+str(args.batch_size)
                                         +"_dim"+str(args.dim)+"_t"+str(args.t)+"_data"+str(args.dataset)+".pth")

    return model

def test_model(model=None): 

    if args.train == False and args.test == True: 
        model = torch.load(args.model_dir)

    if args.dataset == "CIFAR10":
        if args.labels == "1%":
            train_X = np.load('data/cifar_1%_xtrain.npy')
            train_y = np.load('data/cifar_1%_xlabels.npy')
            dataset_train = torch.utils.data.TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_y))
        elif args.labels == "10%":
            train_X = np.load('data/cifar_10%_xtrain.npy')
            train_y = np.load('data/cifar_10%_xlabels.npy')
            dataset_train = torch.utils.data.TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_y))
        else:
            dataset_train = torchvision.datasets.CIFAR10(args.dataset_dir, train = True, download=True, transform=t.TransformsSimCLR().test_transform)
            dataset_test = torchvision.datasets.CIFAR10(args.dataset_dir, train = False, download=True, transform=t.TransformsSimCLR().test_transform)
        linear_model = let.LogisticRegression(model.n_features, 10).cuda()
    elif args.dataset == "IMAGENET":
        data_path = "C:/Users/Dustin/Desktop/simclrv1_plus_v2/imagenet_new/Data/train"
        dataset_train, dataset_test = inet.get_imagenet_datasets(data_path, check_transform=False)
        linear_model = let.LogisticRegression(model.n_features, 1000).cuda()

    else:
        raise NotImplementedError

    # Create Dataloaders
    memory_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=4)
    # Get Encoder
    encoder_f = model.encoder_f
    # Create Training and Test-Set for Regression Model
    (train_X, train_y, test_X, test_y) = let.get_features(encoder_f, memory_loader, test_loader)
    arr_train_loader, arr_test_loader = let.create_data_loaders_from_arrays(train_X, train_y, test_X, test_y, args.batch_size)
    # Set up training
    optimizer = torch.optim.Adam(linear_model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(args.epochs_lineval):
        loss_epoch, accuracy_epoch = let.train_lr(arr_train_loader, linear_model, criterion, optimizer, epoch, args)
        writer.add_scalar("Loss/train_linEval", loss_epoch / len(memory_loader), epoch)
        writer.add_scalar("Accuracy/train_linEval", accuracy_epoch / len(memory_loader), epoch)

    # final testing
    loss_epoch, accuracy_epoch = let.test_lr(arr_test_loader, linear_model, criterion, optimizer)
    writer.add_scalar("Loss/test_linEval", loss_epoch / len(test_loader), epoch)
    writer.add_scalar("Accuracy/test_linEval", accuracy_epoch / len(test_loader), epoch)
    writer.flush()
    print(
        f"[FINAL]\t Loss: {loss_epoch / len(test_loader)}\t Accuracy: {accuracy_epoch / len(test_loader)}"
)

if __name__ == '__main__':
    if args.train == True and args.test == False:
        model = training_model()
    if args.train == True and args.test == True:
        model = training_model()
        test_model(model)
    if args.train == False and args.test == True:
        test_model()