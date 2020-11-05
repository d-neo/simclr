from tqdm import tqdm
import math
import torch
# train for one epoch
import numpy as np
from scipy import ndimage

def mutual_information_2d(x, y, sigma=1, normalized=False):
    """
    Computes (normalized) mutual information between two 1D variate from a
    joint histogram.
    Parameters
    ----------
    x : 1D array
        first variable
    y : 1D array
        second variable
    sigma: float
        sigma for Gaussian smoothing of the joint histogram
    Returns
    -------
    nmi: float
        the computed similariy measure
    """
    bins = (256, 256)
    EPS = np.finfo(float).eps

   # x = x.numpy()
    #y = y.numpy()
    x = x.numpy()
    x = x[0,:,:].ravel()
    y = y.numpy()
    y = y[0,:,:].ravel()
    jh = np.histogram2d(x, y, bins=bins)[0]

    # smooth the jh with a gaussian filter of given sigma
    ndimage.gaussian_filter(jh, sigma=sigma, mode='constant',
                                 output=jh)

    # compute marginal histograms
    jh = jh + EPS
    sh = np.sum(jh)
    jh = jh / sh
    s1 = np.sum(jh, axis=0).reshape((-1, jh.shape[0]))
    s2 = np.sum(jh, axis=1).reshape((jh.shape[1], -1))

    # Normalised Mutual Information of:
    # Studholme,  jhill & jhawkes (1998).
    # "A normalized entropy measure of 3-D medical image alignment".
    # in Proc. Medical Imaging 1998, vol. 3338, San Diego, CA, pp. 132-143.
    if normalized:
        mi = ((np.sum(s1 * np.log(s1)) + np.sum(s2 * np.log(s2)))
                / np.sum(jh * np.log(jh))) - 1
    else:
        mi = ( np.sum(jh * np.log(jh)) - np.sum(s1 * np.log(s1))
               - np.sum(s2 * np.log(s2)))

    return mi


def chooseInfo(im1):
    liste = []
    for j in range(len(im1)):
        summe = 0

        for i in range(im1[j].shape[0]):
            summe += mutual_information_2d(im1[j][i],im1[j][i])

        liste.append(summe/im1[j].shape[0])

 
    index1 = liste.index(min(liste))
    liste.pop(index1)
    index2 = liste.index(min(liste))
    return (index1, index2+1)


def train(net, data_loader, train_optimizer, epoch, args):
    net.train()
    adjust_learning_rate(train_optimizer, epoch, args)

    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    total_neg_sim, total_pos_sim = 0.0, 0.0 
    iterator = 0
    for (images, _) in train_bar:
        train_optimizer.zero_grad()
        i = 2
        j = 1
    # in1, in2 = chooseInfo(images)
    #im_1, im_2 = images[in1].cuda(non_blocking=True), images[in2].cuda(non_blocking=True)
        im_1, im_2 = images[0].cuda(non_blocking=True), images[1].cuda(non_blocking=True)
        loss, neg_sim, pos_sim = net.forward(im_1, im_2, args)

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
        loss.backward()
        train_optimizer.step()
        
        total_neg_sim += neg_sim.item()
        total_pos_sim += pos_sim.item()
        total_num += data_loader.batch_size
        total_loss += loss.item() * data_loader.batch_size
        iterator += 1

        train_bar.set_description('Train Epoch: [{}/{}], lr: {:.6f}, Loss: {:.4f}, Nsim: {:.4f}, Psim: {:.4f}'.format(epoch, args.epochs, train_optimizer.param_groups[0]['lr'], total_loss / total_num, total_neg_sim / iterator, total_pos_sim / iterator))

      
    return (total_loss / total_num, total_neg_sim / iterator, total_pos_sim / iterator)

# lr scheduler for training
def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr