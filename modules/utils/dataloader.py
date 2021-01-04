import glob
from modules.utils.utils import img_to_array
from modules.utils.transformations import TransformsSimCLR
from torchvision import transforms
import numpy as np
import torch
import torchvision
import random
from os import path

class SatelliteDataset(torch.utils.data.Dataset):
    def __init__(self, folder, imgsize=224, transform=None, iterate_over_all=False):
        super(SatelliteDataset).__init__()

        self.imgsize = imgsize

        self.files = glob.glob(folder+'*.tif')
        self.files = [f for f in self.files if "strip_0" not in f]
        self.image = torch.from_numpy(img_to_array(self.files[0], dim_ordering="CHW")).float()
        self.npixels = np.prod(self.image.shape[1:3])
        self.train_transform = transform
        

        self.num_patches = self.npixels // imgsize**2
        self.file_idx = 1
        self.i = 0
        self.iterate_over_all = iterate_over_all

    def __getitem__(self, idx): # getitem(...)
        patch = transforms.RandomCrop(self.imgsize)(self.image)
        if self.train_transform:
            augmented = self.train_transform(patch)
        self.i += 1

        if self.i > self.num_patches and self.iterate_over_all and self.file_idx < len(self.files):
            self.image = torch.from_numpy(img_to_array(self.files[self.file_idx], dim_ordering="CHW")).float()
            #self.file_idx = (self.file_idx + 1) % len(self.files)
            self.file_idx += 1
            self.i = 0

        return (augmented, np.zeros(1))

    def __len__(self):
        if self.iterate_over_all:
            return self.num_patches * len(self.files)
        else:
            return self.num_patches


class SatelliteDatasetLabelled(torch.utils.data.Dataset):
    def __init__(self, transform=None, dset="train"):
        super(SatelliteDatasetLabelled).__init__()

        
        self.trainset = []
        self.valset = []
        self.testset = []
        self.transform = transform
        self.dataset = dset
        threshold = 0.17
        
        for file in glob.glob("../../../data/bangalore/training_data/treecover_segmentation/masks_north/*"):

            filenum = file[78::].split(".")[0]

            src = "../../../data/bangalore/training_data/treecover_segmentation/tiles_north/tile_"+filenum+".tif"
            testsrc =  "../../../data/bangalore/training_data/treecover_segmentation/tiles_north/tiles_on_strip_0/tile_"+filenum+".tif"

            imgArr = img_to_array(file)
            percNonForrest = np.sum(imgArr) / (imgArr.shape[0]*imgArr.shape[1])

            if percNonForrest > threshold: label = 0
            else: label = 1

            if path.exists(testsrc):
                im = torch.from_numpy(img_to_array(testsrc, dim_ordering="CHW")).float()
                self.testset.append((im, label))     
            else:
                im = torch.from_numpy(img_to_array(src, dim_ordering="CHW")).float()
                self.trainset.append((im, label))

        positives = 0
        negatives = 0
        random.seed(0)
        while ((negatives+positives) < 90):
            randint = random.randint(0, len(self.trainset)-1)
            im = self.trainset[randint][0]
            label = self.trainset[randint][1]
            if label == 1 and positives < 45: 
                self.valset.append((im, label))
                positives += 1
                self.trainset.pop(randint)
            if label == 0 and negatives < 45:
                self.valset.append((im, label))
                negatives += 1
                self.trainset.pop(randint)
                    

    def __getitem__(self, idx): 
             
        
        if self.dataset == "train":
            data = self.trainset[idx]
        elif self.dataset == "val":
            data = self.valset[idx]
        else:
            data = self.testset[idx]
            
        if self.transform:
            aug = self.transform(data[0])
            data = (aug, data[1])
            
        return data

    def __len__(self):
        if self.dataset == "train":
            return len(self.trainset)
        elif self.dataset == "val":
            return len(self.valset)
        else:
            return len(self.testset)