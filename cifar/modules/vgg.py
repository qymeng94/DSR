import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from .neuron import LIFNeuron, IFNeuron
from .neuron import rate_spikes, weight_rate_spikes


import torch
cfg = {
    'VGG11': [
        [64, 'M'],
        [128, 'M'],
        [256, 256, 'M'],
        [512, 512, 'M'],
        [512, 512, 'M']
    ],
    'VGG13': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 'M'],
        [512, 512, 'M'],
        [512, 512, 'M']
    ],
    'VGG16': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 256, 'M'],
        [512, 512, 512, 'M'],
        [512, 512, 512, 'M']
    ],
    'VGG19': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 256, 256, 'M'],
        [512, 512, 512, 512, 'M'],
        [512, 512, 512, 512, 'M']
    ]
}


class VGG(nn.Module):
    def __init__(self, snn_setting, vgg_name, num_classes, dropout, neuron_type):
        super(VGG, self).__init__()

        self.timesteps = snn_setting['timesteps']
        self.snn_setting = snn_setting
        self.neuron_type = neuron_type

        self.init_channels = 2
        self.layer1 = self._make_layers(cfg[vgg_name][0], dropout)
        self.layer2 = self._make_layers(cfg[vgg_name][1], dropout)
        self.layer3 = self._make_layers(cfg[vgg_name][2], dropout)
        self.layer4 = self._make_layers(cfg[vgg_name][3], dropout)
        self.layer5 = self._make_layers(cfg[vgg_name][4], dropout)

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        if self.neuron_type == 'lif':
            relu = LIFNeuron(self.snn_setting)
            self.tau = snn_setting['tau']
            self.delta_t = snn_setting['delta_t']
            self.weight_avg = True
        elif self.neuron_type == 'if':
            relu = IFNeuron(self.snn_setting)
            self.weight_avg = False
        else:
            raise NotImplementedError('Please use IF or LIF model.')

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*7*7, num_classes),
            nn.BatchNorm1d(num_classes),
            relu
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, cfg, dropout):
        layers = []
        for x in cfg:
            if x == 'M':
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(self.init_channels, x, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm2d(x))
                if self.neuron_type == 'lif':
                    layers.append(LIFNeuron(self.snn_setting))
                elif self.neuron_type == 'if':
                    layers.append(IFNeuron(self.snn_setting))
                else:
                    raise NotImplementedError('Please use IF or LIF model.')
                layers.append(nn.Dropout(dropout))
                self.init_channels = x
        return nn.Sequential(*layers)

    def forward(self, x):
        inputs = torch.cat([x[:,_,:,:,:] for _ in range(self.timesteps)], 0)
        out = self.layer1(inputs)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.avgpool(out)
        out = self.classifier(out)

        if self.weight_avg:
            out = weight_rate_spikes(out, self.timesteps, self.tau, self.delta_t)
        else:
            out = rate_spikes(out, self.timesteps)
        return out


def vgg11_if(snn_setting, num_classes=10, dropout=0.1):
    return VGG(snn_setting, 'VGG11', num_classes, dropout, 'if')

def vgg11_lif(snn_setting, num_classes=10, dropout=0.1):
    return VGG(snn_setting, 'VGG11', num_classes, dropout, 'lif')
