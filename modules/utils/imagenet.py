from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
from skimage import io
import numpy as np
import time
from PIL import Image

IMG_SIZE = (128,128)

class ImageNetDataset(Dataset):
    def __init__(self, data_path, is_train, train_split = 0.9, random_seed = 42, target_transform = None, num_classes = None, transform=True, numberViews=2):
        super(ImageNetDataset, self).__init__()

        self.numberViews = numberViews
        self.data_path = data_path
        self.transform = transform
        self.is_classes_limited = False
        self.train_transform = transforms.Compose(
            [  
            transforms.RandomResizedCrop(128),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor()
            ]
        )
        self.test_transform = transforms.Compose(
            [
            transforms.ToTensor()
            ]
        )

        if num_classes != None:
            self.is_classes_limited = True
            self.num_classes = num_classes

        self.classes = []
        class_idx = 0
        for class_name in os.listdir(data_path):
            if not os.path.isdir(os.path.join(data_path,class_name)):
                continue
            self.classes.append(
               dict(
                   class_idx = class_idx,
                   class_name = class_name,
               ))
            class_idx += 1

            if self.is_classes_limited:
                if class_idx == self.num_classes:
                    break

        if not self.is_classes_limited:
            self.num_classes = len(self.classes)

        self.image_list = []
        for cls in self.classes:
            class_path = os.path.join(data_path, cls['class_name'])
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                self.image_list.append(dict(
                    cls = cls,
                    image_path = image_path,
                    image_name = image_name,
                ))

        self.img_idxes = np.arange(0,len(self.image_list))

        np.random.seed(random_seed)
        np.random.shuffle(self.img_idxes)

        last_train_sample = int(len(self.img_idxes) * train_split)
        if is_train:
            self.img_idxes = self.img_idxes[:last_train_sample]
        else:
            self.img_idxes = self.img_idxes[last_train_sample:]

    def __len__(self):
        return len(self.img_idxes)

    def __getitem__(self, index):
        img_idx = self.img_idxes[index]
        img_info = self.image_list[img_idx]
        img = Image.open(img_info['image_path'])

        if img.mode == 'L':
            tr = transforms.Grayscale(num_output_channels=3)
            img = tr(img)         
 
        if self.transform==True:  
            return ([self.train_transform(img) for _ in range(self.numberViews)], img_info['cls']['class_idx'])
        else:  
            return (self.test_transform(img), img_info['cls']['class_idx'])

    def get_number_of_classes(self):
        return self.num_classes

    def get_number_of_samples(self):
        return self.__len__()

    def get_class_names(self):
        return [cls['class_name'] for cls in self.classes]

    def get_class_name(self, class_idx):
        return self.classes[class_idx]['class_name']

def get_imagenet_datasets(data_path, num_classes = None, test = True, check_transform=True, numberViews=2):
    random_seed = int(time.time())
    dataset_train = ImageNetDataset(data_path,is_train = True, random_seed=random_seed, num_classes = num_classes, transform=check_transform, numberViews=numberViews)
    if test == True:
        dataset_test = ImageNetDataset(data_path, is_train = False, random_seed=random_seed, num_classes = num_classes, transform=check_transform)
        return dataset_train, dataset_test

    return dataset_train