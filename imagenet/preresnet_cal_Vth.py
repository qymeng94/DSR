from __future__ import division

import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class DataStatus():
    def __init__(self, max_num=1e7):
        self.pool = []
        self.num = 0
        self.max_num = max_num

    def append(self, data):
        self.pool.append(data.view(-1))
        self.num += self.pool[-1].size()[0]
        if self.num > self.max_num:
            self.random_shrink()

    def random_shrink(self):
        tensor = torch.cat(self.pool, 0)
        tensor = tensor[torch.randint(len(tensor), size=[int(self.max_num // 2)])]
        self.pool.clear()
        self.pool.append(tensor)

    def max(self, fraction=1, relu=True, max_num=1e6):
        tensor = torch.cat(self.pool, 0)
        if len(tensor) > max_num:
            tensor = tensor[torch.randint(len(tensor), size=[int(max_num)])]
        if relu:
            tensor = F.relu(tensor)
        if fraction == 1:
            return tensor.max()
        else:
            tensor_sort = tensor.sort()[0]
            return tensor_sort[int(fraction * tensor_sort.size(0))]


class PreBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(PreBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample

        self.storein1 = DataStatus()
        self.storein2 = DataStatus()

    def forward(self, x):
        temp = self.bn1(x)
        self.storein1.append(temp.detach().cpu())
        x = self.relu1(temp)
        out = self.conv1(x)

        out = self.bn2(out)
        self.storein2.append(out.detach().cpu())
        out = self.relu2(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out += residual
        return out


class PreBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(PreBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.downsample = downsample

        self.storein1 = DataStatus()
        self.storein2 = DataStatus()
        self.storein3 = DataStatus()

    def forward(self, x):
        temp = self.bn1(x)
        self.storein1.append(temp.detach().cpu())
        x = self.relu1(temp)

        out = self.conv1(x)

        out = self.bn2(out)
        self.storein2.append(out.detach().cpu())
        out = self.relu2(out)
        out = self.conv2(out)

        out = self.bn3(out)
        self.storein3.append(out.detach().cpu())
        out = self.relu3(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out += residual

        return out

class PreResNet(nn.Module):
    def __init__(self, bottleneck=True, baseWidth=64, head7x7=True, layers=(3, 4, 23, 3), num_classes=1000):
        """ Constructor
        Args:
            layers: config of layers, e.g., (3, 4, 23, 3)
            num_classes: number of classes
        """
        super(PreResNet, self).__init__()
        block = PreBasicBlock if not bottleneck else PreBottleneck

        self.inplanes = baseWidth  # default 64

        self.conv1 = nn.Conv2d(3, baseWidth, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(baseWidth)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, baseWidth, layers[0])
        self.layer2 = self._make_layer(block, baseWidth * 2, layers[1], 2)
        self.layer3 = self._make_layer(block, baseWidth * 4, layers[2], 2)
        self.layer4 = self._make_layer(block, baseWidth * 8, layers[3], 2)
        self.bn2 = nn.BatchNorm2d(baseWidth * 8 * block.expansion)
        self.relu2 = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc = nn.Linear(baseWidth * 8 * block.expansion, num_classes)

        self.bn3 = nn.BatchNorm1d(num_classes)
        self.relu4 = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.storein1 = DataStatus()
        self.storein2 = DataStatus()
        self.storein3 = DataStatus()
        self.storein4 = DataStatus()
        self.num_blocks = layers

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

    def get_max_activation(self):
        act = {'storein1': self.storein1.max(fraction=0.999).detach().cpu(),
               'storein2': self.storein2.max(fraction=0.999).detach().cpu(),
               'storein3': self.storein3.max(fraction=0.999).detach().cpu(),
               'storein4': self.storein4.max(fraction=0.999).detach().cpu(),
               }


        lens1 = len([1] + [1] * (self.num_blocks[0] - 1))
        lens2 = len([1] + [1] * (self.num_blocks[1] - 1))
        lens3 = len([1] + [1] * (self.num_blocks[2] - 1))
        lens4 = len([1] + [1] * (self.num_blocks[3] - 1))

        for j in ['1', '2', '3', '4']:
            for i in range(eval('lens' + j)):
                act['layer' + j + '[' + str(i) + ']' + '.storein1'] = eval(
                    'self.layer' + j + '[' + str(i) + '].storein1.max(fraction=0.999).detach().cpu(),')

                act['layer' + j + '[' + str(i) + ']' + '.storein2'] = eval(
                    'self.layer' + j + '[' + str(i) + '].storein2.max(fraction=0.999).detach().cpu(),')

                if 'conv3' in dir(self.layer1[0]):
                    act['layer' + j + '[' + str(i) + ']' + '.storein3'] = eval(
                        'self.layer' + j + '[' + str(i) + '].storein3.max(fraction=0.999).detach().cpu(),')

        print(act)
        return act

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        self.storein1.append(x.detach().cpu())
        x = self.relu1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn2(x)
        self.storein2.append(x.detach().cpu())
        x = self.relu2(x)

        x = self.avgpool(x)
        self.storein3.append(x.detach().cpu())
        x = self.relu3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        x = self.bn3(x)
        self.storein4.append(x.detach().cpu())
        x = self.relu4(x)

        return x


def resnet18(snn_setting, num_classes=1000):
    model = PreResNet(bottleneck=False, baseWidth=64, head7x7=True, layers=(2, 2, 2, 2), num_classes=num_classes)
    return model


def preresnet50(snn_setting, num_classes=1000):
    model = PreResNet(bottleneck=True, baseWidth=64, head7x7=True, layers=(3, 4, 6, 3), num_classes=num_classes)
    return model
