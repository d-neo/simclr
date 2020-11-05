import torchvision.models as models
import torch.nn as nn

def get_resnet(name, pretrained=False):
    resnets = {
        "resnet18": models.resnet18(pretrained=pretrained),
        "resnet50": models.resnet50(pretrained=pretrained),
    }
    if name not in resnets.keys():
        raise KeyError(f"{name} is not a valid ResNet version")
    return resnets[name]

def modify_resnet_model(model):
    conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    nn.init.kaiming_normal_(conv1.weight, mode='fan_out', nonlinearity='relu')
    model.conv1 = conv1
    model.maxpool = nn.Identity()
    return model