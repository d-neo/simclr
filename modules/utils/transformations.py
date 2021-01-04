from torchvision import transforms
from PIL import Image, ImageOps 
from PIL import ImageFilter, ImageEnhance
import random 
import math
import numpy as np 
import torchvision
import torch

######################################
# Possible Transformations ############
######################################
class GaussianBlur(object):
    """Gaussian blur augmentation"""
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class MySolarizeTransform(object):
    """Solarizing augmentation"""
    def __init__(self, t):
        self.t = t
    
    def __call__(self, x):
        return ImageOps.solarize(x, threshold = self.t)
    
class GaussianNoiseOld(object):
    """Gaussian Noise augmentation"""
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, inputs):
        mean = 0
        stddev = 0.01
        ins = inputs.cpu()
        input_array = ins.data.numpy()

        noise = np.random.normal(loc=mean, scale=stddev, size=np.shape(input_array))
        
        out = np.add(input_array, noise)
        output_tensor = torch.from_numpy(out)
        out = output_tensor.float()
        
        return out    

class Equalize():
    """Equalize augmentation"""
    def __call__(self, x):
        return ImageOps.equalize(x)
    
class AutoContrast():
    """Autocontrast augmentation"""
    def __call__(self, x):
        return ImageOps.autocontrast(x)

class Invert():
    """Invert augmentation"""
    def __call__(self, x):
        return ImageOps.invert(x)
     
class Posterize():
    """Posterize augmentation"""
    def __call__(self, x):
        return ImageOps.posterize(x, 2)

class Color():
    """Color augmentation"""
    def __call__(self, x):
        return ImageEnhance.Color(x).enhance(2)

class Sharpness():
    """Sharpness augmentation"""
    def __call__(self, x):
        return ImageEnhance.Sharpness(x).enhance(2)



######################################
# Transformations - MultiChannel ############
######################################

class multiChannelGrayscale():
    def __call__(self, x):
        #if x.shape[0] == 7:
        #out1 = torch.mean(x, dim=1).float().unsqueeze(-1).reshape(x.shape[0],1, 224, 224)
       # else:
        out1 = torch.mean(x, dim=0).float().unsqueeze(-1).reshape(1, 224, 224)
        out = torch.cat([out1, out1], 0) 
        for _ in range(5):
            out = torch.cat([out, out1], 0) 
        return out

class multiChannelRotate():
    def __call__(self,x):
        return transforms.functional.rotate(x, angle=90.0)

class GaussianNoise(object):
    def __init__(self, center=0, std=50):
        self.center = center
        self.std = std

    def __call__(self, X):
        noise = np.random.normal(self.center, self.std, X.shape)
        X = X + noise
        return X.float()

class Vignetting(object):
    def __init__(self,
                 ratio_min_dist=0.2,
                 range_vignette=(0.2, 0.8),
                 random_sign=False):
        self.ratio_min_dist = ratio_min_dist
        self.range_vignette = np.array(range_vignette)
        self.random_sign = random_sign

    def __call__(self, X):
        h, w = X.shape[:2]
        min_dist = np.array([h, w]) / 2 * np.random.random() * self.ratio_min_dist

        # create matrix of distance from the center on the two axis
        x, y = np.meshgrid(np.linspace(-w/2, w/2, w), np.linspace(-h/2, h/2, h))
        x, y = np.abs(x), np.abs(y)

        # create the vignette mask on the two axis
        x = (x - min_dist[0]) / (np.max(x) - min_dist[0])
        x = np.clip(x, 0, 1)
        y = (y - min_dist[1]) / (np.max(y) - min_dist[1])
        y = np.clip(y, 0, 1)

        # then get a random intensity of the vignette
        vignette = (x + y) / 2 * np.random.uniform(*self.range_vignette)
        vignette = np.tile(vignette[..., None], [1, 1, 224])

        sign = 2 * (np.random.random() < 0.5) * (self.random_sign) - 1
        X = X * (1 + sign * vignette)

        return X.float()

class Contrast(object):
    def __init__(self, range_contrast=(-50, 50)):
        self.range_contrast = range_contrast

    def __call__(self, X):
        contrast = np.random.randint(*self.range_contrast)
        X = X * (contrast / 127 + 1) - contrast
        return X.float()

class Brightness(object):
    def __init__(self, range_brightness=(-50, 50)):
        self.range_brightness = range_brightness

    def __call__(self, X):
        brightness = np.random.randint(*self.range_brightness)
        X = X + brightness
        return X.float()

######################################
# Transformation - Module ############
######################################
class TransformsSimCLR:
    """
    Input:
        imgsize: type int - Size of the Image
        mean: type List(int) - Mean of the Dataset
        std: type List(int) - Standard Deviation of the Dataset

    A stochastic data augmentation module that transforms any given data example randomly 
    resulting in x correlated views of the same example, which we consider as a positive pair.
    """

    def __init__(self, imgsize, mean=None, std=None, numberViews=2):

        self.numberViews = numberViews

        self.train_transform = torchvision.transforms.Compose(
            [  
            transforms.RandomResizedCrop(imgsize),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.8, 0.8, 0.8, 0.2) 
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur()], p=0.5),
            transforms.ToTensor(),
            #torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )

        self.test_transform = torchvision.transforms.Compose(
            [
            torchvision.transforms.ToTensor(),
            #torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )


    def __call__(self, x):
        return [self.train_transform(x) for _ in range(self.numberViews)]



class TransformsSimCLR_SAT:
    """
    Input:
        imgsize: type int - Size of the Image
        mean: type List(int) - Mean of the Dataset
        std: type List(int) - Standard Deviation of the Dataset

    A stochastic data augmentation module that transforms any given data example randomly 
    resulting in x correlated views of the same example, which we consider as a positive pair.
    """
    def __init__(self, imgsize, mean=None, std=None, numberViews=2):
        self.numberViews = numberViews

        self.train_transform_SAT_UNSUP = torchvision.transforms.Compose(
            [
            transforms.RandomResizedCrop(imgsize),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([multiChannelRotate()], p=0.5),
            transforms.RandomApply([Brightness(), Contrast()], p=0.8),
            transforms.RandomApply([Vignetting()], p=0.3),
            #transforms.RandomApply([GaussianNoise()], p=0.5),
            transforms.RandomApply([multiChannelGrayscale()], p=0.2)
            ]
        )

        self.train_transform_SAT_SUP = torchvision.transforms.Compose(
            [
            transforms.RandomResizedCrop(imgsize),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([multiChannelRotate()], p=0.5),
            transforms.RandomApply([Brightness(), Contrast()], p=0.8),
            transforms.RandomApply([Vignetting()], p=0.3),
            #transforms.RandomApply([GaussianNoise()], p=0.5),
            transforms.RandomApply([multiChannelGrayscale()], p=0.2)
            ]
        )

        self.test_transform_SAT_SUP = torchvision.transforms.Compose(
            [
            transforms.Resize(imgsize)
            ]
        )

    def __call__(self, x):
        return [self.train_transform_SAT_UNSUP(x) for _ in range(self.numberViews)]

