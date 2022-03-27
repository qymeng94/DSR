from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import numpy as np
import math
import collections
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from modules import preresnet, vgg
from utils.cifar10_dvs import CIFAR10DVS

parser = argparse.ArgumentParser(description='PyTorch SNN Training')
# Basic settings
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('--path', default='./data', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--model', default='preresnet.resnet18_if', type=str)
parser.add_argument('--name', default='', type=str)
# SNN settings
parser.add_argument('--timesteps', default=20, type=int)
parser.add_argument('--Vth', default=0.3, type=float)
parser.add_argument('--tau', default=1.0, type=float)
parser.add_argument('--delta_t', default=0.05, type=float)
parser.add_argument('--alpha', default=0.3, type=float)
parser.add_argument('--train_Vth', default=1, type=int)
parser.add_argument('--Vth_bound', default=0.0005, type=float)
parser.add_argument('--rate_stat', default=0, type=int)
# Optimization options
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=128, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=50, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--warmup', default=400, type=int)
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--nesterov', action='store_true')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--optimizer', default='SGD', type=str, help='which optimizer')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='./checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pin_memory', action='store_true')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'

#FP16
scaler = torch.cuda.amp.GradScaler()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy
current_iter = 0


def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch
    args.T_max = args.epochs

    if not os.path.isdir(args.checkpoint + '/' + args.dataset):
        mkdir_p(args.checkpoint + '/' + args.dataset)

    # Data
    print('==> Preparing dataset %s' % args.dataset)
    assert args.dataset == 'cifar10' or args.dataset == 'cifar100' or args.dataset == 'CIFAR10DVS', 'Dataset can only be cifar10, cifar100 or CIFAR10DVS.'
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        if args.dataset == 'cifar10':
            dataloader = datasets.CIFAR10
            num_classes = 10
            data_normalization = [0.4914, 0.4822, 0.4465, 0.2023, 0.1994, 0.2010]
        elif args.dataset == 'cifar100':
            dataloader = datasets.CIFAR100
            num_classes = 100
            data_normalization = [0.5071, 0.4867, 0.4408, 0.2675, 0.2565, 0.2761]

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((data_normalization[0], data_normalization[1], data_normalization[2]),
                                 (data_normalization[3], data_normalization[4], data_normalization[5])),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((data_normalization[0], data_normalization[1], data_normalization[2]),
                                 (data_normalization[3], data_normalization[4], data_normalization[5])),
        ])

        trainset = dataloader(root=args.path, train=True, download=True, transform=transform_train)
        trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)
        testset = dataloader(root=args.path, train=False, download=False, transform=transform_test)
        testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)


    elif args.dataset == 'CIFAR10DVS':

        num_classes = 10
        transform_train = transforms.Compose([
            transforms.Resize([48, 48]),
            transforms.RandomCrop(48, padding=4),
        ])
        trainset = CIFAR10DVS(args.path, train=True, split_ratio=0.9, use_frame=True,
                              frames_num=args.timesteps,
                              split_by='number', normalization=None, transform=transform_train)
        trainloader = data.DataLoader(
            dataset=trainset,
            batch_size=args.train_batch,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=args.pin_memory,
            drop_last=False)

        transform_test = transforms.Compose([
            transforms.Resize([48, 48]),
        ])

        testset = CIFAR10DVS(args.path, train=False, split_ratio=0.9, use_frame=True,
                             frames_num=args.timesteps,
                             split_by='number', normalization=None, transform=transform_test)
        testloader = data.DataLoader(
            dataset=testset,
            batch_size=args.test_batch,
            num_workers=args.workers,
            pin_memory=args.pin_memory,
            drop_last=False)

    # Model
    print("==> creating model")
    snn_setting = {}
    snn_setting['timesteps'] = args.timesteps
    snn_setting['train_Vth'] = True if args.train_Vth == 1 else False
    snn_setting['Vth'] = args.Vth
    snn_setting['tau'] = args.tau
    snn_setting['delta_t'] = args.delta_t
    snn_setting['alpha'] = args.alpha
    snn_setting['Vth_bound'] = args.Vth_bound
    snn_setting['rate_stat'] = True if args.rate_stat == 1 else False

    model = eval(args.model + '(snn_setting,num_classes=' + str(num_classes) + ')')
    if use_cuda:
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))

    # Resume
    title = 'cifar-' + 'SNN_Conv'
    if args.resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(
            args.checkpoint + '/' + args.dataset + '/' + args.resume), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.checkpoint + '/' + args.dataset + '/' + args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        try:
            logger = Logger(os.path.join(args.checkpoint, args.dataset, args.model+args.name + '.txt'), title=title, resume=True)
        except:
            print('Cannot open the designated log file, have created a new one.')
            logger = Logger(os.path.join(args.checkpoint, args.dataset, args.model + args.name + '.txt'), title=title)
            logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])
    else:
        logger = Logger(os.path.join(args.checkpoint, args.dataset, args.model+args.name + '.txt'), title=title)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])



    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(testloader, model, criterion)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        logger.append([0.0, 0.0, test_loss, 0.0, test_acc])
        logger.close()
        try:
            firing_rate = model.cal_rate()
            torch.save(firing_rate, os.path.join(args.checkpoint, args.dataset, args.model + args.name + '_firing_rate.dict'))
        except:
            try:
                firing_rate = model.module.cal_rate()
                torch.save(firing_rate,os.path.join(args.checkpoint, args.dataset, args.model + args.name + '_firing_rate.dict'))
            except:
                print('Cannot calculate the firing rate.')
        return


    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc = train(trainloader, model, criterion, optimizer, warmup=args.warmup)
        test_loss, test_acc = test(testloader, model, criterion)

        # append logger file
        logger.append([state['lr'], train_loss, test_loss, train_acc, test_acc])

        ## save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': test_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint=args.checkpoint+'/'+args.dataset, filename=args.model+args.name)

    logger.close()

    print('Best acc:')
    print(best_acc)

    logger.plot()
    savefig(os.path.join(args.checkpoint, args.dataset, args.model+args.name + '.pdf'))


def train(trainloader, model, criterion, optimizer, warmup=0):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(trainloader))

    global current_iter

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if warmup != 0 and current_iter < warmup:
            adjust_warmup_lr(optimizer, current_iter, warmup)
            current_iter += 1
        # measure data loading time
        data_time.update(time.time() - end)

        inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
            batch=batch_idx + 1,
            size=len(trainloader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)


def test(testloader, model, criterion):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(testloader))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
            batch=batch_idx + 1,
            size=len(testloader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint'):
    filepath = os.path.join(checkpoint, filename+'.pth')
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, filename+'_best.pth'))



def adjust_learning_rate(optimizer, epoch):
    global state
    state['lr'] = 0.5 * args.lr * (1 + math.cos(epoch/args.T_max*math.pi))
    for param_group in optimizer.param_groups:
        param_group['lr'] = state['lr']


def adjust_warmup_lr(optimizer, citer, warmup):
    for param_group in optimizer.param_groups:
        param_group['lr'] = state['lr'] * (citer + 1.) / warmup

if __name__ == '__main__':
    whole_time = time.time()
    main()
    print('Whole running time:', time.time() - whole_time)
