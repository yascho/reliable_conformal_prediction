import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.resnet import ResNet18_Weights, ResNet50_Weights
from torchvision.models.resnet import ResNet101_Weights


def create_image_classifier(hparams, dataset_mean, dataset_std):
    arch = hparams['arch']

    if not hparams['pretrained']:
        if arch == "ResNet101":
            model = ResNet(Bottleneck, [3, 4, 23, 3],
                           in_channels=hparams["in_channels"],
                           out_channels=hparams["out_channels"])
        elif arch == "ResNet50":
            model = ResNet(Bottleneck,
                           num_blocks=[3, 4, 6, 3],
                           in_channels=hparams["in_channels"],
                           out_channels=hparams["out_channels"])
        elif arch == "ResNet18":
            model = ResNet(BasicBlock,
                           num_blocks=[2, 2, 2, 2],
                           in_channels=hparams["in_channels"],
                           out_channels=hparams["out_channels"])
        else:
            raise Exception("Not implemented")
    else:
        if arch == "ResNet101":
            model = torchvision.models.resnet101(
                weights=ResNet101_Weights.DEFAULT
            )
        elif arch == "ResNet50":
            model = torchvision.models.resnet50(
                weights=ResNet50_Weights.DEFAULT
            )
        elif arch == "ResNet18":
            model = torchvision.models.resnet18(
                weights=ResNet18_Weights.DEFAULT
            )
        else:
            raise Exception("Not implemented")

        if model.fc.out_features != hparams['out_channels']:
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, hparams['out_channels'])

    normalize = NormalizeLayer(dataset_mean, dataset_std)
    model = torch.nn.Sequential(normalize, model)

    return model.to(hparams["device"])


class ResNet(nn.Module):
    """
        ResNet implementation from https://github.com/alevine0/randomizedAblation/blob/master/pytorch_cifar/models/resnet_unmodified.py

        ResNet paper:
            - Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
            Deep Residual Learning for Image Recognition. CVPR 2016.
    """

    def __init__(self, block, num_blocks, in_channels=3, out_channels=10):
        # in_channels in [3,4,6]
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, out_channels)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class NormalizeLayer(torch.nn.Module):
    """
      Normalization layer, code adapted from (Cohen et al., 2019):
      https://github.com/locuslab/smoothing
    """

    def __init__(self, means, stds):
        super(NormalizeLayer, self).__init__()
        self.means = means
        self.stds = stds

    def forward(self, input, batched=True):

        if batched:
            (batch_size, num_channels, height, width) = input.shape
            means = self.means.repeat(
                (batch_size, height, width, 1)).permute(0, 3, 1, 2)
            stds = self.stds.repeat(
                (batch_size, height, width, 1)).permute(0, 3, 1, 2)
        else:
            (num_channels, height, width) = input.shape
            means = self.means.repeat((height, width, 1)).permute(2, 0, 1)
            stds = self.stds.repeat((height, width, 1)).permute(2, 0, 1)

        return (input - means.to(input.device)) / stds.to(input.device)
