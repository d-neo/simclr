# Import own modules
# SelfSupervised
import modules.selfsupervised.simclr_model as sm 
import modules.selfsupervised.simclr_train as smtr
# Evaluation
import modules.evaluation.lin_eval_testmoduls as let
# Utils
import modules.utils.transformations as t 
import modules.utils.loss_functions as lf
import modules.utils.lars as lars
# Settings
import modules.utils.get_data as get 
# Reporting 
import modules.utils.reporting as report 
# Saving Model
import modules.utils.save_model as savem
# Get Model
import modules.utils.get_model as gm 
# Import extern modules
import argparse
import torchvision
import torch 
import torch.nn as nn
from torch.nn.parallel import DataParallel
import math 
import numpy as np

from torch.optim import lr_scheduler 

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
parser.add_argument('--cos', default="False", type=str, help='use cosine lr schedule')
parser.add_argument('--squared', default="False", type=str, help='use square lr schedule')
parser.add_argument('--batch-size', default=768, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--wd', default=1e-6, type=float, metavar='W', help='weight decay')
parser.add_argument('--dataset', default="IMAGENET", type=str, metavar='W', help='dataset')
parser.add_argument('--dataset-dir', default="data/imagenet128", type=str, metavar='W', help='dataset directory')
parser.add_argument('--optimizer', default="adam", type=str, metavar='W', help='Adam, SGD, Lars')
# SIMCLR specific configs:
parser.add_argument('--dim', default=128, type=int, help='feature dimension')
parser.add_argument('--t', default=0.1, type=float, help='softmax temperature')
parser.add_argument('--numberviews', default=2, type=int, help='Number of Views created for Avg.Loss')
# Training/Test Settings
parser.add_argument('--model-dir', default="", type=str, metavar='W', help='if test true give model dir!')
parser.add_argument('--model-name', default="", type=str, metavar='W', help='if test true give model dir!')
# Linear Evaluation
parser.add_argument('--epochs-lineval', default=100, type=int, metavar='N', help='number of total epochs to run in linEval')
# Labels
parser.add_argument('--labels', default="full", type=str, metavar='N', help='1%, 10% or full labelled data sets are supported')
# Multiple GPUs
parser.add_argument('--dataparallel', default="True", type=str, metavar='W', help='multiple gpus true or false')
# Reload
parser.add_argument('--reload', default="False", type=str, metavar='W', help='reload true or false')
# Verbose
parser.add_argument('--verbose', default="True", type=str, help='True if you want to evaluate your representations after each epoch (obviously SLOWER!)')

# Set up args
args = parser.parse_args()

# Convert 
args.dataparallel = True if str.lower(args.dataparallel) == "true" else False
args.reload = True if str.lower(args.reload) == 'true' else False
args.verbose = True if str.lower(args.verbose) == 'true' else False
args.cos = True if str.lower(args.cos) == 'true' else False
args.squared = True if str.lower(args.squared) == 'true' else False
args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args.num_gpus = torch.cuda.device_count()

def training_model(): 
    """

    """
    # Set Reporting up
    writer_train, writer_val, mkdirpath = report.setReportingUp_trainingSimclr(args)
    # Get Data
    train_loader, train_loader_nonaug, val_loader = get.data_trainingSimclr(args)
    # Get encoder
    encoder = get.encoder(args)
    args.featureDim, args.numberClasses, epoch_start = encoder.fc.in_features, get.numberClasses(args), 1
    # Choose Model
    model = sm.modelSIMCLR(encoder, args.featureDim, lf.contrastive_loss_cosine_extra, dim=args.dim, T=args.t) # lf.contrastive_loss_cosine_extra is standard
    # To device
    model = model.to(args.device)
    # Get Optimizer
    optimizer = get.optimizer_training(model, args)
    # Check Reload
    if args.reload: 
        checkpoint = torch.load("saved_models/selfsupervised/"+args.model_dir+"/"+args.model_name, map_location=args.device.type)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_start = checkpoint['epoch']+1
    # Check Dataparallel
    if args.dataparallel: model = DataParallel(model)

    scheduler = None

    # Start training
    for epoch in range(epoch_start, args.epochs + 1):
        train_loss, neg_sim, pos_sim, lr = smtr.train(model, train_loader, optimizer, epoch, args)
        if scheduler: 
            scheduler.step()
        if args.verbose:
            train_loss_eval, val_loss_eval, train_acc, val_acc = let.featureEval(model, train_loader_nonaug, val_loader, epoch, args, extended=False, maxepochs_nonextended=100)
            writer_train, writer_val = report.updateWriter_trainingSimclr(writer_train, writer_val, train_loss_eval, val_loss_eval, train_acc, 
                                                                            val_acc, pos_sim, neg_sim, train_loss, lr, epoch)
        else: 
            writer_train.add_scalar("Similarity/train", pos_sim, epoch)
            writer_val.add_scalar("Similarity/train", neg_sim, epoch)
            writer_train.add_scalar("LR_Contrastive/train", lr, epoch)
            writer_train.add_scalar("Loss_Contrastive/train", train_loss, epoch)
        if (epoch % 5) == 0: savem.save_model(model, optimizer, mkdirpath, epoch)

    savem.save_model(model, optimizer, mkdirpath, epoch)
    # Longer Evaluation on last feature representation
    _, _, train_acc_long, val_acc_long = let.featureEval(model, train_loader_nonaug, val_loader, epoch, args, extended=True)
    if args.verbose:
        writer_train = report.updateWriter_trainingSimclr_extended(writer_train, train_loss, train_acc, val_acc, train_acc_long, val_acc_long, args)
    else: 
        writer_train = report.updateWriter_trainingSimclr_extended(writer_train, train_loss, train_acc_long, val_acc_long, train_acc_long, val_acc_long, args)

    return

if __name__ == '__main__':
    training_model()
