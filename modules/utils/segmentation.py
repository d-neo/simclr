
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torchvision
from torchvision import transforms
from torch import nn
from torch.nn import functional as nnf

import numpy as np 
from tqdm import tqdm
import math
import glob
from PIL import Image
from os import path
import random
from statistics import mean 

from . import utils as ut
from . import dataloader as satelliteLoader

def plot(title, label, train_results, val_results, yscale='linear', save_path=None, 
         extra_pt=None, extra_pt_label=None):
    """Plot learning curves.

    Args:
        title (str): Title of plot
        label (str): x-axis label
        train_results (list): Results vector of training of length of number
            of epochs trained. Could be loss or accuracy.
        val_results (list): Results vector of validation of length of number
            of epochs. Could be loss or accuracy.
        yscale (str, optional): Matplotlib.pyplot.yscale parameter. 
            Defaults to 'linear'.
        save_path (str, optional): If passed, figure will be saved at this path.
            Defaults to None.
        extra_pt (tuple, optional): Tuple of length 2, defining x and y coordinate
            of where an additional black dot will be plotted. Defaults to None.
        extra_pt_label (str, optional): Legend label of extra point. Defaults to None.
    """
    
    epoch_array = np.arange(len(train_results)) + 1
    train_label, val_label = "Training "+label.lower(), "Validation "+label.lower()
    
    sns.set(style='ticks')

    plt.plot(epoch_array, train_results, epoch_array, val_results, linestyle='dashed', marker='o')
    legend = ['Train results', 'Validation results']
    
    if extra_pt:
        ####################
        ## YOUR CODE HERE ##
        ####################
        extra_point=plt.plot(extra_pt[0],extra_pt[1],'k.',label=extra_pt_label, markersize=16)
        if extra_pt_label:
            legend.append(extra_pt_label)

        # END OF YOUR CODE #
        
    plt.legend(legend)
    plt.xlabel('Epoch')
    plt.ylabel(label)
    plt.yscale(yscale)
    plt.title(title)
    
    sns.despine(trim=True, offset=5)
    plt.title(title, fontsize=15)
    if save_path:
        plt.savefig(str(save_path), bbox_inches='tight')
    plt.show()


class DecoderBlock(nn.Module):

    def __init__(self, input_channels, output_channels):
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels

        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1)

    def forward(self, x, skip_x):
        x_up = nnf.interpolate(x, size=(skip_x.size(2), skip_x.size(3)), mode='bilinear', align_corners=False)
        x = torch.cat([x_up, skip_x], dim=1)
        x = self.conv(x)
        x = nnf.relu(x)
        return x


class ResNetSegmentation(nn.Module):
    """
    decoder shape defines different factors for the number of maps in the decoder layers. Use small resnet only for RN18.
    """

    def __init__(self, base_network, decoder='m', outputs=10, small_resnet=False):
        super().__init__()

        self.resnet = base_network

        if hasattr(self.resnet, 'fc'):
            del self.resnet.fc

        if type(decoder) == str:
            self.dec_sizes = {
                'xs': (32, 32, 16, 16, 8),
                's': (256, 128, 64, 16, 16),
                'm': (256, 128, 64, 32, 32),
                'l': (16, 128, 64, 48, 48)
            }[decoder]
        elif type(decoder) in {list, tuple}:
            self.dec_sizes = decoder
        else:
            raise ValueError('invalid decoder configuration')

        if hasattr(self.resnet, 'sizes') and small_resnet:
            self.enc_sizes = self.resnet.sizes[::-1] + tuple([base_network.inputs])
        elif hasattr(self.resnet, 'sizes') and not small_resnet:
            self.enc_sizes = [x*4 for x in self.resnet.sizes[::-1]] + [base_network.inputs]
        elif not small_resnet:
            self.enc_sizes = [2048, 1024, 512, 256, base_network.inputs]
        else:
            self.enc_sizes = [512, 256, 128, 64, base_network.inputs]

        print('enc', self.enc_sizes)
        print('dec', self.dec_sizes)

        self.decoder2 = DecoderBlock(self.enc_sizes[0] + self.enc_sizes[1], self.dec_sizes[0])
        self.decoder3 = DecoderBlock(self.dec_sizes[0] + self.enc_sizes[2], self.dec_sizes[1])
        self.decoder4 = DecoderBlock(self.dec_sizes[1] + self.enc_sizes[3], self.dec_sizes[2])
        self.decoder6 = DecoderBlock(self.dec_sizes[2] + self.enc_sizes[4], self.dec_sizes[4])
        self.post_conv = nn.Conv2d(self.dec_sizes[4], outputs, (1, 1))

    def forward(self, x):

        layer_seq = [['conv1', 'bn1', 'relu', 'maxpool', 'layer1'], ['layer2'], ['layer3'], ['layer4']]

        x0 = x
        activations = []
        for layers in layer_seq:
            for layer in layers:
                x = getattr(self.resnet, layer)(x)

            activations += [x]

        decoder_activations = []
        for i, layer in enumerate(['decoder2', 'decoder3', 'decoder4']):
            x = getattr(self, layer)(x, activations[-i-2])
            decoder_activations += [x]

        x = self.decoder6(x, x0)
        x = self.post_conv(x)

        return x

class SatelliteDatasetLabelled(torch.utils.data.Dataset):
    def __init__(self, transform=None, dset="train"):
        super(SatelliteDatasetLabelled).__init__()

        
        self.trainset = []
        self.valset = []
        self.testset = []
        self.transform = transform
        self.dataset = dset
        
        for file in glob.glob("../../../data/bangalore/training_data/treecover_segmentation/masks_north/*"):
            filenum = file[78::].split(".")[0]

            src = "../../../data/bangalore/training_data/treecover_segmentation/tiles_north/tile_"+filenum+".tif"
            testsrc =  "../../../data/bangalore/training_data/treecover_segmentation/tiles_north/tiles_on_strip_0/tile_"+filenum+".tif"

            imgArr = ut.img_to_array(file)
        
            if path.exists(testsrc):
                im = torch.from_numpy(ut.img_to_array(testsrc, dim_ordering="CHW")).float()
                imgArr = torch.from_numpy(ut.img_to_array(file, dim_ordering="CHW")).float()
                self.testset.append((im, imgArr))     
            else:
                im = torch.from_numpy(ut.img_to_array(src, dim_ordering="CHW")).float()
                imgArr = torch.from_numpy(ut.img_to_array(file, dim_ordering="CHW")).float()
                self.trainset.append((im, imgArr))

        i = 0
        random.seed(0)
        while i < 90:
            randint = random.randint(0, len(self.trainset)-1)
            im = self.trainset[randint][0]
            mask = self.trainset[randint][1]
            self.valset.append((im, mask))
            self.trainset.pop(randint)
            i += 1 

    def __getitem__(self, idx): 
             
        
        if self.dataset == "train":
            data = self.trainset[idx]
        elif self.dataset == "val":
            data = self.valset[idx]
        else:
            data = self.testset[idx]
            
        if self.transform:
            aug = self.transform(data[0])
            aug1 = self.transform(data[1])
            data = (aug, aug1)
            
        return data

    def __len__(self):
        if self.dataset == "train":
            return len(self.trainset)
        elif self.dataset == "val":
            return len(self.valset)
        else:
            return len(self.testset)
