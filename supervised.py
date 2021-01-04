# Import own modules
# Utils
import modules.utils.transformations as t 
import modules.utils.loss_functions as lf
import modules.utils.lars as lars
# Saving Model
import modules.utils.save_model as savem
# Get Model
import modules.utils.get_model as gm 
# Reporting 
import modules.utils.reporting as report 
# Settings
import modules.utils.get_data as get 
# Evaluation
import modules.evaluation.lin_eval_testmoduls as let
# Import extern modules
import argparse
import torchvision
import torch 
import torch.nn as nn
from torch.nn.parallel import DataParallel
from tqdm import tqdm
import math 
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np

# DELETE LATER
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Create an Argparser
parser = argparse.ArgumentParser(description='Train simCLRv1 on CIFAR-10')
# Model-Architecture
parser.add_argument('-a', '--arch', default='resnet18')
# lr: 0.06 for batch 512 (or 0.03 for batch 256) 0.12 for 1024?
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--epochs', default=20, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--schedule', default=[1020, 1060], nargs='*', type=int, help='learning rate schedule (when to drop lr by 10x); does not take effect if --cos is on')
parser.add_argument('--cos', default="False" , type=str, help='use cosine lr schedule')
parser.add_argument('--squared', default="False", type=str, help='use square lr schedule')
parser.add_argument('--batch-size', default=256, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--wd', default=1e-6, type=float, metavar='W', help='weight decay')
parser.add_argument('--dataset', default="IMAGENET", type=str, metavar='W', help='dataset')
parser.add_argument('--dataset-dir', default="data/imagenet128", type=str, metavar='W', help='dataset directory')
parser.add_argument('--optimizer', default="adam", type=str, metavar='W', help='Adam, SGD, Lars')

# Training/Test Settings
parser.add_argument('--model-dir', default="", type=str, metavar='W', help='if test true give model dir!')
parser.add_argument('--model-name', default="", type=str, metavar='W', help='if test true give model dir!')
# Labels
parser.add_argument('--labels', default="full", type=str, metavar='N', help='1%, 10% or full labelled data sets are supported')
# Multiple GPUs
parser.add_argument('--dataparallel', default="True", type=str, metavar='W', help='multiple gpus true or false')
# Reload
parser.add_argument('--reload', default="False", type=str, metavar='W', help='reload true or false')
parser.add_argument('--reload-epoch', default=1, type=int, metavar='W', help='epoch the reloaded model was saved on')

# Set up args
args = parser.parse_args()

# Convert 
args.dataparallel = True if str.lower(args.dataparallel) == "true" else False
args.reload = True if str.lower(args.reload) == 'true' else False
args.cos = True if str.lower(args.cos) == 'true' else False
args.squared = True if str.lower(args.squared) == 'true' else False
args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args.num_gpus = torch.cuda.device_count()

def training_model(): 
    writer_train, writer_val, mkdirpath = report.setReportingUp_trainingSupervised(args)
    # Get Data
    train_loader, val_loader = get.data_trainingSupervised(args)
    # Get encoder/model in the supervised case
    model = get.encoder(args)
    # Choose Model
    if args.arch == "efficientnet-b0":
        model._fc = torch.nn.Linear(model._fc.in_features, get.numberClasses(args))
    else:
        model.fc = torch.nn.Linear(model.fc.in_features, get.numberClasses(args))

     # To Device
    model = model.to(args.device)

    # Set Optimizer
    optimizer = get.optimizer_training(model, args)

    criterion = nn.CrossEntropyLoss()
    epoch_start = 1
    # Reload
    if args.reload: 
        checkpoint = torch.load("saved_models/supervised/"+args.model_dir+"/"+args.model_name, map_location=args.device.type)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_start = checkpoint['epoch']+1
    # Set Dataparallel
    if args.dataparallel: model = DataParallel(model)
    # Start training
    for epoch in range(epoch_start, args.epochs + 1):
        train_loss, train_acc = let.train_lr(train_loader, model, criterion, optimizer, epoch, args.epochs, args)
        #train_loss, train_acc, lr = st.train(model, train_loader, optimizer, criterion, epoch, args)
        val_loss, val_acc = let.test_lr(val_loader, model, criterion, epoch, args.epochs, args)
        #val_loss, val_acc = st.val(model, val_loader, criterion, epoch, args)
        writer_train, writer_val = report.updateWriter_trainingSupervised(writer_train, writer_val, train_loss, val_loss, train_acc, val_acc, epoch)
        if (epoch % 10) == 0: savem.save_model(model, optimizer, mkdirpath, epoch)
    writer_train = report.updateWriter_trainingSupervised_extended(writer_train, train_loss, train_acc, val_acc, args)
    savem.save_model(model, optimizer, mkdirpath, epoch)

    return

if __name__ == '__main__':
    training_model()