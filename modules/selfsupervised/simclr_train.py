from tqdm import tqdm
import math
import torch
import numpy as np
from scipy import ndimage
from statistics import mean 

def train(net, data_loader, train_optimizer, epoch, args):
    """
    Input:
        net: The model you want to train
        data_loader: Pytorch dataloader containing the data
        train_optimizer: The optimizer used during training
        epoch: Type int - the epoch the model is currently trained
        args: Argument list

        Training the SIMCLR model - includes average loss (denoted as a comment in code).
    """
    net.train()
    #adjust_learning_rate(train_optimizer, epoch, args)
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    total_neg_sim, total_pos_sim = 0.0, 0.0 
    iterator = 0

    for (images, _) in train_bar:
        train_optimizer.zero_grad()
        i = 2
        j = 1
        im_1, im_2 = images[0].cuda(non_blocking=True), images[1].cuda(non_blocking=True)
        loss, neg_sim, pos_sim = net.forward(im_1, im_2, args)

        # Average-Loss
        while(i<len(images)):
            im_1, im_2 = images[i].cuda(non_blocking=True), images[i+1].cuda(non_blocking=True)
            loss_iter, neg_sim_iter, pos_sim_iter = net.forward(im_1, im_2, args)
            loss += loss_iter
            neg_sim += neg_sim_iter
            pos_sim += pos_sim_iter
            i += 2
            j += 1

        loss /= j 
        neg_sim /= j
        pos_sim /= j

        if args.dataparallel: 
            loss.mean().backward()
            total_neg_sim += neg_sim.mean().item()
            total_pos_sim += pos_sim.mean().item()
            total_loss += loss.mean().item() * data_loader.batch_size
        else:
            loss.backward()
            total_neg_sim += neg_sim.item()
            total_pos_sim += pos_sim.item()
            total_loss += loss.item() * data_loader.batch_size

        train_optimizer.step()
        total_num += data_loader.batch_size

        iterator += 1
        train_bar.set_description('Train Epoch: [{}/{}], lr: {:.6f}, Loss_Contrast: {:.4f}, Nsim: {:.4f}, Psim: {:.4f}'.format(epoch, args.epochs, train_optimizer.param_groups[0]['lr'], total_loss / total_num, total_neg_sim / iterator, total_pos_sim / iterator))

      
    return (total_loss / total_num, total_neg_sim / iterator, total_pos_sim / iterator, train_optimizer.param_groups[0]['lr'])


def adjust_learning_rate(optimizer, epoch, args):
    """
    Input:
        optimizer: Optimizer used for training
        epoch: Type int - the epoch the model is currently training in
        args: Argument list 

        Decay the learning rate based on schedule.
    """
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    elif args.squared:
        lr = lr/(epoch**0.5)
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr