# SimCLR
PyTorch implementation of SimCLR: A Simple Framework for Contrastive Learning of Visual Representations by T. Chen et al.
Including support for:
- Data parallel training
- LARS (Layer-wise Adaptive Rate Scaling) optimizer

[Link to paper](https://arxiv.org/pdf/2002.05709.pdf)

### Results
These are the top-1 accuracy of linear classifiers trained on the (frozen) representations learned by SimCLR:

| Method  | Batch Size | ResNet | Projection output dimensionality | Epochs | Optimizer | CIFAR-10 | ImageNet (128x128)
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| SimCLR + Linear eval. | 256 | ResNet18 | 128 | 100 | Adam | 0.83 | 0.35 | 
| AVGSimCLR + Linear eval. | 256 | ResNet50 | 128 | 100 | Adam | 0.861 | 0.356 | 
| SimCLR + Finetuning (100% labels) | 256 | ResNet18 | 128 | 100 |  Adam | 0.904  | 0.438 |
| AVGSimCLR + Finetuning (100% labels) | 256 | ResNet18 | 128 | 40 | Adam | 0.915  | 0.443 |
| Logistic Regression | - | - | - | 40 | Adam | 0.358 | 0.389 |


#### LARS optimizer
The LARS optimizer is implemented in `modules/utils/lars.py`. It can be activated by setting the --optimizer parameter to "lars". It is still experimental and has not been thoroughly tested.

## What is SimCLR?
SimCLR is a "simple framework for contrastive learning of visual representations". The contrastive prediction task is defined on pairs of augmented examples, resulting in 2N examples per minibatch. Two augmented versions of an image are considered as a correlated, "positive" pair (x_i and x_j). The remaining 2(N - 1) augmented examples are considered negative examples. The contrastive prediction task aims to identify x_j in the set of negative examples for a given x_i.

<p align="center">
  <img src="https://github.com/Spijkervet/SimCLR/blob/master/media/architecture.png?raw=true" width="500"/>
</p>

## Logging and TensorBoard
To view results in TensorBoard, run:
```
tensorboard --logdir runs
```

## Optimizers and learning rate schedule
This implementation features the Adam optimizer and the LARS optimizer, with the option to decay the learning rate using a cosine decay schedule. The optimizer and weight decay can be configured via the parameters --optimizer and --wd.

#### Dependencies
```
torch
torchvision
tensorboard
```
