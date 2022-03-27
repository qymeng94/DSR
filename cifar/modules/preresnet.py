import torch.nn as nn
import torch
from .neuron import LIFNeuron, IFNeuron
from .neuron import rate_spikes, weight_rate_spikes


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_channels, out_channels, stride, dropout, snn_setting, neuron_type):
        super(PreActBlock, self).__init__()
        self.timesteps = snn_setting['timesteps']
        self.bn1 = nn.BatchNorm2d(in_channels)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size=3, stride=1, padding=1)

        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, padding=0)
        else:
            self.shortcut = nn.Sequential()

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
        out = self.conv2(self.dropout(self.relu2(self.bn2(out))))
        out = out + self.shortcut(x)
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_channels, out_channels, stride, dropout, snn_setting, neuron_type):
        super(PreActBottleneck, self).__init__()
        self.timesteps = snn_setting['timesteps']
        self.bn1 = nn.BatchNorm2d(in_channels)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv3 = nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size=1, stride=1, padding=0)

        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, padding=0)
        else:
            self.shortcut = nn.Sequential()

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
        out = self.conv2(self.relu2(self.bn2(out)))
        out = self.conv3(self.dropout(self.relu3(self.bn3(out))))

        out = out + self.shortcut(x)

        return out


class PreActResNet(nn.Module):

    def __init__(self, block, num_blocks, num_classes, dropout, snn_setting, neuron_type):
        super(PreActResNet, self).__init__()
        self.neuron_type = neuron_type
        self.timesteps = snn_setting['timesteps']
        self.num_blocks = num_blocks

        self.init_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], 1, dropout, snn_setting)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], 2, dropout, snn_setting)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], 2, dropout, snn_setting)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], 2, dropout, snn_setting)

        self.bn1 = nn.BatchNorm2d(512 * block.expansion)
        self.pool = nn.AvgPool2d(4)
        self.flat = nn.Flatten()
        self.drop = nn.Dropout(dropout)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        self.bn2 = nn.BatchNorm1d(num_classes)

        if neuron_type == 'lif':
            self.relu1 = LIFNeuron(snn_setting)
            self.relu2 = LIFNeuron(snn_setting)
            self.relu3 = LIFNeuron(snn_setting)
            self.weight_avg = True
            self.tau = snn_setting['tau']
            self.delta_t = snn_setting['delta_t']
        elif neuron_type == 'if':
            self.relu1 = IFNeuron(snn_setting)
            self.relu2 = IFNeuron(snn_setting)
            self.relu3 = IFNeuron(snn_setting)
            self.weight_avg = False
        else:
            raise NotImplementedError('Please use IF or LIF model.')


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def _make_layer(self, block, out_channels, num_blocks, stride, dropout, snn_setting):
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.init_channels, out_channels, stride, dropout, snn_setting, self.neuron_type))
            self.init_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.cat([x for _ in range(self.timesteps)], 0)
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.relu2(self.pool(self.relu1(self.bn1(out))))
        out = self.linear(self.drop(self.flat(out)))
        out = self.relu3(self.bn2(out))

        if self.weight_avg:
            out = weight_rate_spikes(out, self.timesteps, self.tau, self.delta_t)
        else:
            out = rate_spikes(out, self.timesteps)
        return out

    def cal_rate(self):
        firing_rate = {'relu1': self.relu1.firing_rate.avg().detach().cpu().numpy(),
               'relu2': self.relu2.firing_rate.avg().detach().cpu().numpy(),
               'relu3': self.relu3.firing_rate.avg().detach().cpu().numpy(),
               }

        lens1 = len([1] + [1] * (self.num_blocks[0] - 1))
        lens2 = len([1] + [1] * (self.num_blocks[1] - 1))
        lens3 = len([1] + [1] * (self.num_blocks[2] - 1))
        lens4 = len([1] + [1] * (self.num_blocks[3] - 1))

        for j in ['1', '2', '3', '4']:
            for i in range(eval('lens' + j)):
                firing_rate['layer' + j + '[' + str(i) + ']' + '.relu1'] = eval(
                    'self.layer' + j + '[' + str(i) + '].relu1.firing_rate.avg().detach().cpu().numpy()')
                firing_rate['layer' + j + '[' + str(i) + ']' + '.relu2'] = eval(
                    'self.layer' + j + '[' + str(i) + '].relu2.firing_rate.avg().detach().cpu().numpy()')
                try:
                    firing_rate['layer' + j + '[' + str(i) + ']' + '.relu3'] = eval(
                        'self.layer' + j + '[' + str(i) + '].relu3.firing_rate.avg().detach().cpu().numpy()')
                except:
                    pass
        #print(firing_rate)
        return firing_rate


def resnet18_lif(snn_setting, num_classes=100,  dropout=0):
    return PreActResNet(PreActBlock, [2,2,2,2], num_classes, dropout, snn_setting, 'lif')

def resnet18_if(snn_setting, num_classes=100,  dropout=0):
    return PreActResNet(PreActBlock, [2,2,2,2], num_classes, dropout, snn_setting, 'if')

