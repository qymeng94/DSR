from __future__ import division
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch
from neuron import LIFNeuron, IFNeuron
from neuron import rate_spikes, weight_rate_spikes



def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class PreBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, snn_setting, neuron_type, inplanes, planes, stride=1, downsample=None):
        super(PreBasicBlock, self).__init__()
        self.timesteps = snn_setting['timesteps']
        self.conv1 = conv3x3(inplanes, planes, stride)

        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.bn2 = nn.BatchNorm2d(planes)

        if neuron_type == 'lif':
            self.relu1 = LIFNeuron(snn_setting)
            self.relu2 = LIFNeuron(snn_setting)
        elif neuron_type == 'if':
            self.relu1 = IFNeuron(snn_setting)
            self.relu2 = IFNeuron(snn_setting)
        else:
            raise NotImplementedError('Please use IF or LIF model.')


    def forward(self, x):
        x = self.relu1(self.bn1(x))
        out = self.conv1(x)

        out = self.bn2(out)
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

    def __init__(self, snn_setting, neuron_type, inplanes, planes, stride=1, downsample=None):
        super(PreBottleneck, self).__init__()
        self.timesteps = snn_setting['timesteps']
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.downsample = downsample

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.bn3 = nn.BatchNorm2d(planes)

        if neuron_type == 'lif':
            self.relu1 = LIFNeuron(snn_setting)
            self.relu2 = LIFNeuron(snn_setting)
            self.relu3 = LIFNeuron(snn_setting)
        elif neuron_type == 'if':
            self.relu1 = IFNeuron(snn_setting)
            self.relu2 = IFNeuron(snn_setting)
            self.relu3 = IFNeuron(snn_setting)
        else:
            raise NotImplementedError('Please use IF or LIF model.')

    def forward(self, x):
        x = self.relu1(self.bn1(x))
        out = self.conv1(x)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu3(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out += residual

        return out

class PreResNet(nn.Module):
    def __init__(self, snn_setting, neuron_type, bottleneck=True, baseWidth=64, layers=(3, 4, 23, 3), num_classes=1000):

        super(PreResNet, self).__init__()
        self.snn_setting = snn_setting
        self.neuron_type = neuron_type
        self.timesteps = snn_setting['timesteps']

        self.num_blocks = layers
        block = PreBasicBlock if not bottleneck else PreBottleneck
        self.inplanes = baseWidth  # default 64
        self.conv1 = nn.Conv2d(3, baseWidth, 7, 2, 3, bias=False)
        self.maxpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, baseWidth, layers[0])
        self.layer2 = self._make_layer(block, baseWidth * 2, layers[1], 2)
        self.layer3 = self._make_layer(block, baseWidth * 4, layers[2], 2)
        self.layer4 = self._make_layer(block, baseWidth * 8, layers[3], 2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(baseWidth * 8 * block.expansion, num_classes)

        self.bn1 = nn.BatchNorm2d(baseWidth)
        self.bn2 = nn.BatchNorm2d(baseWidth * 8 * block.expansion)
        self.bn3 = nn.BatchNorm1d(num_classes)

        if neuron_type == 'lif':
            self.relu1 = LIFNeuron(snn_setting)
            self.relu2 = LIFNeuron(snn_setting)
            self.relu3 = LIFNeuron(snn_setting)
            self.relu4 = LIFNeuron(snn_setting)
        elif neuron_type == 'if':
            self.relu1 = IFNeuron(snn_setting)
            self.relu2 = IFNeuron(snn_setting)
            self.relu3 = IFNeuron(snn_setting)
            self.relu4 = IFNeuron(snn_setting)
        else:
            raise NotImplementedError('Please use IF or LIF model.')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if neuron_type == 'lif':
            self.weight_avg = True
            self.tau = snn_setting['tau']
            self.delta_t = snn_setting['delta_t']
        else:
            self.weight_avg = False

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.snn_setting, self.neuron_type, self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.snn_setting, self.neuron_type, self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.cat([x for _ in range(self.timesteps)], 0)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.relu2(self.bn2(x))
        x = self.relu3(self.avgpool(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.relu4(self.bn3(x))

        if self.weight_avg:
            out = weight_rate_spikes(x, self.timesteps, self.tau, self.delta_t)
        else:
            out = rate_spikes(x, self.timesteps)
        return out





def resnet18_if(snn_setting, num_classes=1000):
    model = PreResNet(snn_setting, 'if', bottleneck=False, baseWidth=64, layers=(2, 2, 2, 2), num_classes=num_classes)
    return model

def resnet34_if(snn_setting, num_classes=1000):
    model = PreResNet(snn_setting, 'if', bottleneck=False, baseWidth=64, layers=(3, 4, 6, 3), num_classes=num_classes)
    return model


def resnet50_if(snn_setting, num_classes=1000):
    model = PreResNet(snn_setting, 'if', bottleneck=True, baseWidth=64, layers=(3, 4, 6, 3), num_classes=num_classes)
    return model



def resnet18_lif(snn_setting, num_classes=1000):
    model = PreResNet(snn_setting, 'lif', bottleneck=False, baseWidth=64, layers=(2, 2, 2, 2), num_classes=num_classes)
    return model

def resnet34_lif(snn_setting, num_classes=1000):
    model = PreResNet(snn_setting, 'lif', bottleneck=False, baseWidth=64, layers=(3, 4, 6, 3), num_classes=num_classes)
    return model


def resnet50_lif(snn_setting, num_classes=1000):
    model = PreResNet(snn_setting, 'lif', bottleneck=True, baseWidth=64, layers=(3, 4, 6, 3), num_classes=num_classes)
    return model

