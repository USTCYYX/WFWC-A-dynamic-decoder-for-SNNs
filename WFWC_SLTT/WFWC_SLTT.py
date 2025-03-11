import datetime
import os
import time
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
from models import spiking_resnet_imagenet, spiking_resnet, spiking_vgg_bn
from modules import neuron
import argparse
from spikingjelly.clock_driven import functional
from spikingjelly.clock_driven import surrogate as surrogate_sj
from modules import surrogate as surrogate_self
from utils import Bar, Logger, AverageMeter, accuracy
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchtoolbox.transform import Cutout
from utils.augmentation import ToPILImage, Resize, ToTensor
import collections
import random
import numpy as np
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main():

    parser = argparse.ArgumentParser(description='WFWC_SLTT')
    parser.add_argument('-T', default=4, type=int, help='simulating time-steps')
    parser.add_argument('-c', default=6, type=int, help='WC threshold')
    parser.add_argument('-r', default=0.5, type=float, help='WR threshold')
    parser.add_argument('-j', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('-data_dir', type=str, default='./data', help='directory of the used dataset')
    parser.add_argument('-dataset', default='cifar10', type=str, help='should be cifar10, cifar100, or imagenet')
    parser.add_argument('-model', type=str, default='spiking_resnet18', help='use which SNN model')
    parser.add_argument('-device', default='cuda:0', help='device')    
    parser.add_argument('-tau', default=1.1, type=float, help='a hyperparameter for the LIF model')
    parser.add_argument('-surrogate', default='triangle', type=str, help='used surrogate function. should be sigmoid, rectangle, or triangle')
    parser.add_argument('-drop_rate', type=float, default=0.0, help='dropout rate. ')

    args = parser.parse_args()
    print(args)


    ########################################################
    # data preparing
    ########################################################
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        c_in = 3
        if args.dataset == 'cifar10':
            dataloader = datasets.CIFAR10
            num_classes = 10
            normalization_mean = (0.4914, 0.4822, 0.4465)
            normalization_std = (0.2023, 0.1994, 0.2010)
        elif args.dataset == 'cifar100':
            dataloader = datasets.CIFAR100
            num_classes = 100
            normalization_mean = (0.5071, 0.4867, 0.4408)
            normalization_std = (0.2675, 0.2565, 0.2761)


        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(normalization_mean, normalization_std),
        ])


        testset = dataloader(root=args.data_dir, train=False, download=False, transform=transform_test)
        test_data_loader = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=args.j)
    elif args.dataset == 'imagenet':
        num_classes = 1000
        traindir = os.path.join(args.data_dir, 'train')
        valdir = os.path.join(args.data_dir, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        test_data_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=1, shuffle=False,
            num_workers=args.j, pin_memory=True)
    else:
        raise NotImplementedError


    ##########################################################
    # model preparing
    ##########################################################
    if args.surrogate == 'sigmoid':
        surrogate_function = surrogate_sj.Sigmoid()
    elif args.surrogate == 'rectangle':
        surrogate_function = surrogate_self.Rectangle()
    elif args.surrogate == 'triangle':
        surrogate_function = surrogate_sj.PiecewiseQuadratic()
        
    neuron_model = neuron.SLTTNeuron

    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        net = spiking_resnet.__dict__[args.model](neuron=neuron_model, num_classes=num_classes, neuron_dropout=args.drop_rate,
                                                  tau=args.tau, surrogate_function=surrogate_function, c_in=c_in, fc_hw=1)
        print('using Resnet model.')
    elif args.dataset == 'imagenet':
        net = spiking_resnet_imagenet.__dict__[args.model](neuron=neuron_model, num_classes=num_classes, neuron_dropout=args.drop_rate,
                                                           tau=args.tau, surrogate_function=surrogate_function, c_in=3)
        print('using NF-Resnet model.')
    else:
        raise NotImplementedError
    print('Total Parameters: %.2fM' % (sum(p.numel() for p in net.parameters()) / 1000000.0))
    net.to(args.device)
    if args.dataset == 'cifar10':
        model=torch.load('./model/checkpoint_cifar10.pth',map_location=torch.device(args.device))
    elif args.dataset == 'cifar100':
        model=torch.load('./model/checkpoint_cifar100.pth',map_location=torch.device(args.device))
    elif args.dataset == 'imagenet':
        model=torch.load('./model/checkpoint_imagenet.pth',map_location=torch.device(args.device))
        
    net.load_state_dict(model["net"]) 


        ############### testing ###############
    net.eval()
    test_acc = 0
    test_samples = 0 #number of test samples
    time=0. #total time steps for test samples
    with torch.no_grad():
        for img, label in test_data_loader:
            voltage_change=0.
            voltage_total=0.
            max_old=0.
            max_new=0.
            counter=0.
            flag=0.
            img = img.float().to(args.device)
            label = label.to(args.device)
            for t in range(args.T):
                voltage_change = net(img)
                voltage_total=voltage_total+voltage_change
                max_new=voltage_total.argmax(dim=1)
                if t==0 :
                    top10_relu=torch.where(torch.topk(voltage_total,10).values>0.,torch.topk(voltage_total,10).values,0.)
                    top10_relu_quadratic_sum=top10_relu.norm(2) ** 2
                    rate=(voltage_total.max() ** 2)/top10_relu_quadratic_sum
                    if rate>args.r:
                        result=max_new
                        time=time+1
                        flag=1.
                        break
                    else:
                        max_old=max_new
                        counter=counter+1
                else:
                    if (max_new==max_old) and (voltage_change.argmax(dim=1) == max_old):
                        counter=counter+1
                        if counter>=args.c :    
                            result=max_new
                            time=time+t+1
                            flag=1.
                            break
                    else:
                        counter=1.
                        max_old=max_new    
                                        
            if flag==0.:
                result= voltage_total.argmax(dim=1)
                time=time+args.T                  
            test_acc += (result == label).float().sum().item()
            test_samples += label.numel()
            functional.reset_net(net)
    test_acc /= test_samples
    print('test accuracy: ',test_acc)
    print('total time steps for test samples: ',time)
    print('number of test samples: ',test_samples)
    print('avarage time steps: ',time/test_samples)


if __name__ == '__main__':
    main()
