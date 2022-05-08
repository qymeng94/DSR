import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.distributed as dist
from typing import Optional, Any
from torch import Tensor

def rate_spikes(data, timesteps):
    chw = data.size()[1:]
    firing_rate = torch.mean(data.view(timesteps, -1, *chw), 0)
    return firing_rate


def weight_rate_spikes(data, timesteps, tau, delta_t):
    chw = data.size()[1:]
    data_reshape = data.view(timesteps, -1, *chw).permute(list(range(1,len(chw)+2)) + [0])
    weight = torch.tensor([math.exp(-1/tau*(delta_t * timesteps-ii*delta_t)) for ii in range(1, timesteps+1)]).to(data_reshape.device)
    return (weight * data_reshape).sum(dim=len(chw)+1) / weight.sum()


def sum_tensor(tensor: torch.Tensor) -> torch.Tensor:
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt


class IFFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, timesteps=10, Vth=1.0, alpha=0.5):
        ctx.save_for_backward(input)

        chw = input.size()[1:]
        input_reshape = input.view(timesteps, -1, *chw)
        mem_potential = torch.zeros(input_reshape.size(1), *chw).to(input_reshape.device)
        spikes = []

        for t in range(input_reshape.size(0)):
            mem_potential = mem_potential + input_reshape[t]
            spike = ((mem_potential >= alpha * Vth).float() * Vth).float()
            mem_potential = mem_potential - spike
            spikes.append(spike)
        output = torch.cat(spikes, 0)

        ctx.timesteps = timesteps
        ctx.Vth = Vth
        return output

    @staticmethod
    def backward(ctx, grad_output):
        with torch.no_grad():
            input = ctx.saved_tensors[0]
            timesteps = ctx.timesteps
            Vth = ctx.Vth

            input_rate_coding = rate_spikes(input, timesteps)
            grad_output_coding = rate_spikes(grad_output, timesteps) * timesteps

            input_grad = grad_output_coding.clone()
            input_grad[(input_rate_coding < 0) | (input_rate_coding > Vth)] = 0
            input_grad = torch.cat([input_grad for _ in range(timesteps)], 0) / timesteps

            Vth_grad = grad_output_coding.clone()
            Vth_grad[input_rate_coding <= Vth] = 0
            Vth_grad = torch.sum(Vth_grad)
            if torch.cuda.device_count() != 1:
                Vth_grad = sum_tensor(Vth_grad)

            return input_grad, None, Vth_grad, None


class RateStatus(nn.Module):
    '''
    Record the average firing rate of one neuron layer.
    '''
    def __init__(self, max_num=1e6):
        super().__init__()
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

    def avg(self, max_num=1e6):
        tensor = torch.cat(self.pool, 0)
        if len(tensor) > max_num:
            tensor = tensor[torch.randint(len(tensor), size=[int(max_num)])]
        return tensor.mean()


class IFNeuron(nn.Module):
    def __init__(self, snn_setting):
        super().__init__()
        self.timesteps = snn_setting['timesteps']
        if snn_setting['train_Vth']:
            self.Vth = nn.Parameter(torch.tensor(snn_setting['Vth']))
        else:
            self.Vth = torch.tensor(snn_setting['Vth'])
        self.alpha = snn_setting['alpha']
        self.Vth_bound = snn_setting['Vth_bound']
        self.rate_stat = snn_setting['rate_stat']
        if self.rate_stat:
            self.firing_rate = RateStatus()

    def forward(self, x):
        with torch.no_grad():
            self.Vth.copy_(F.relu(self.Vth - self.Vth_bound) + self.Vth_bound)
        iffunc = IFFunction.apply
        out = iffunc(x, self.timesteps, self.Vth, self.alpha)
        if not self.training and self.rate_stat:
            with torch.no_grad():
                self.firing_rate.append(rate_spikes(out, self.timesteps) / self.Vth)
        return out




def generate_spike_lif(out_s,mem_potential, Vth=0.1, tau=1.0, delta_t=0.05, alpha=0.3):
    beta = torch.exp(-delta_t/tau)
    spikes = []
    for t in range(out_s.size(0)):
        mem_potential = beta * mem_potential + (1 - beta) * out_s[t]
        spike = ((mem_potential >= alpha * Vth).float() * Vth).float()
        mem_potential -= spike
        spikes.append(spike / delta_t)
    return spikes



def lif_grad(Vth, delta_t, grad_output_coding, input_rate_coding, tau, timesteps):
    indexes = (input_rate_coding > 0) & (input_rate_coding < Vth / delta_t * tau)

    input_grad = torch.zeros_like(grad_output_coding)
    input_grad[indexes] = grad_output_coding[indexes].clone() / tau

    Vth_grad = grad_output_coding.clone()
    Vth_grad[input_rate_coding <= Vth / delta_t * tau] = 0
    Vth_grad = torch.sum(Vth_grad) * delta_t
    if torch.cuda.device_count() != 1:
        Vth_grad = sum_tensor(Vth_grad)

    input_grad = torch.cat([input_grad for _ in range(timesteps)], 0) / timesteps
    return input_grad, Vth_grad


class LIFFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, timesteps, Vth, tau, delta_t=0.05, alpha=0.3):
        ctx.save_for_backward(input)

        chw = input.size()[1:]
        input_reshape = input.view(timesteps, -1, *chw)
        mem_potential = torch.zeros(input_reshape.size(1), *chw).to(input_reshape.device)
        beta = torch.exp(-delta_t / tau)

        spikes = []
        for t in range(input_reshape.size(0)):
            mem_potential = beta * mem_potential + (1 - beta) * input_reshape[t]
            spike = ((mem_potential >= alpha * Vth).float() * Vth).float()
            mem_potential = mem_potential - spike
            spikes.append(spike / delta_t)

        output = torch.cat(spikes, 0)

        ctx.timesteps = timesteps
        ctx.Vth = Vth
        ctx.tau = tau
        ctx.delta_t = delta_t
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        timesteps = ctx.timesteps
        Vth = ctx.Vth
        delta_t = ctx.delta_t
        tau = ctx.tau

        input_rate_coding = weight_rate_spikes(input, timesteps, tau, delta_t)
        grad_output_coding = weight_rate_spikes(grad_output, timesteps, tau, delta_t) * timesteps
        input_grad, Vth_grad = lif_grad(Vth, delta_t, grad_output_coding, input_rate_coding, tau, timesteps)

        return input_grad, None, Vth_grad, None, None, None



class LIFNeuron(nn.Module):
    def __init__(self, snn_setting):
        super().__init__()
        self.timesteps = snn_setting['timesteps']
        if snn_setting['train_Vth']:
            # if train_Vth is True, then Vth and tau are set to be trainable. default:True
            self.Vth = nn.Parameter(torch.tensor(snn_setting['Vth']))
        else:
            self.Vth = torch.tensor(snn_setting['Vth'])
        self.tau = torch.tensor(snn_setting['tau'])
        self.delta_t = snn_setting['delta_t']
        self.alpha = snn_setting['alpha']
        self.Vth_bound = snn_setting['Vth_bound']
        self.rate_stat = snn_setting['rate_stat']
        if self.rate_stat:
            self.firing_rate = RateStatus()

    def forward(self, x):
        with torch.no_grad():
            self.Vth.copy_(F.relu(self.Vth - self.Vth_bound) + self.Vth_bound)

        lif = LIFFunction.apply
        out = lif(x, self.timesteps, self.Vth, self.tau, self.delta_t, self.alpha)
        if not self.training and self.rate_stat:
            with torch.no_grad():
                self.firing_rate.append(rate_spikes(out, self.timesteps) / self.Vth * self.delta_t)
        return out
