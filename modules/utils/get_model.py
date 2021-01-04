import torchvision.models as models
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

"""
def get_resnet(name, pretrained=False):
    resnets = {
        "resnet18": models.resnet18(pretrained=pretrained),
        "resnet50": models.resnet50(pretrained=pretrained),
        "efficientnet-b0": EfficientNet.from_name('efficientnet-b0')
    }
    if name not in resnets.keys():
        raise KeyError(f"{name} is not a valid ResNet version")
    return resnets[name]
"""


def get_resnet(name, pretrained=False, bn_momentum=0.1):
    """
    Input:
        name: type String - the name of the model you want to import from pytorch
        (only resnet18, resnet 50 supported)

        Loads either a resnet50 or resnet18 from pytorch
    """
    if name == "resnet18":
        net = models.resnet18(pretrained=pretrained, norm_layer=lambda x: nn.BatchNorm2d(x, momentum=bn_momentum))

    elif name == "resnet50":
        net = models.resnet50(pretrained=pretrained, norm_layer=lambda x: nn.BatchNorm2d(x, momentum=bn_momentum))

    elif name == "efficientnet-b0":
        net = EfficientNet.from_name('efficientnet-b0')

    else:
        raise NotImplementedError

    return net


def modify_resnet_model(model):
    """
    Input:
        model: The model you want to modify.

        Modifies model for CIFAR10 dataset (32x32 pixels).
    """
    conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    nn.init.kaiming_normal_(conv1.weight, mode='fan_out', nonlinearity='relu')
    model.conv1 = conv1
    model.maxpool = nn.Identity()
    return model


def modify_resnet_model_SATELLITE(model, nbands=7):
    """
    Input:
        model: The model you want to modify.

        Modifies model for SATELLITE dataset (7 Channels).
    """
    model.conv1 = nn.Conv2d(nbands, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)    # 224x224
        x = self.bn1(x)     
        x = self.relu(x)
        x = self.maxpool(x)  # 112x112

        x = self.layer1(x)   # 56x56
        x = self.layer2(x)   # 28x28
        x = self.layer3(x)   # 14x14
        x = self.layer4(x)   # 7x7

        x = self.avgpool(x)  # 1x1
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet34(**kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model