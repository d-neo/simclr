from torchvision import transforms
from PIL import Image, ImageOps 
from PIL import ImageFilter
import random 
import math
import numpy as np 
import torchvision
import torch

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class MySolarizeTransform(object):
    def __init__(self, t):
        self.t = t
    
    def __call__(self, x):
        return ImageOps.solarize(x, threshold = self.t)
    
class GaussianNoise(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

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


class TransformsSimCLR:
    """
    A stochastic data augmentation module that transforms any given data example randomly 
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self):
        self.train_transform = torchvision.transforms.Compose(
            [  
            transforms.RandomResizedCrop(32),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            #transforms.RandomApply([MySolarizeTransform(130)], p=0.2),
            transforms.ToTensor(),
            #transforms.RandomApply([GaussianNoise([.1, 2.])], p=0.5),
            ]
        )

        self.test_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor()
            ]
        )

    def __call__(self, x):
        return [self.train_transform(x), self.train_transform(x)]
    